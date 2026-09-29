"""Exercise the same immutable publisher used by Jenkins without AWS access."""

import hashlib
import io
import runpy
from pathlib import Path
from unittest.mock import Mock
from zipfile import ZipFile, ZipInfo

import pytest
from botocore.exceptions import ClientError

publisher = runpy.run_path(str(Path(__file__).resolve().parents[2] / "ci" / "publish_wheel.py"))
publish_wheel = publisher["publish_wheel"]


def wheel(members=None, year=2020):
    output = io.BytesIO()
    if members is None:
        members = [("package.py", b"code"), ("metadata", b"version")]
    with ZipFile(output, "w") as archive:
        for name, body in members:
            archive.writestr(ZipInfo(name, (year, 1, 1, 0, 0, 0)), body)
    return output.getvalue()


def client(stored=None, error=None):
    s3 = Mock()
    s3.get_object.return_value = {"Body": io.BytesIO(wheel() if stored is None else stored)}
    if error:
        s3.put_object.side_effect = ClientError({"Error": {"Code": error}}, "PutObject")
    return s3


def test_new_artifact_is_conditionally_written_and_verified():
    body = wheel()
    s3 = client(body)
    assert publish_wheel(s3, "bucket", "key", body) == hashlib.sha256(body).hexdigest()
    s3.put_object.assert_called_once_with(
        Bucket="bucket",
        Key="key",
        Body=body,
        IfNoneMatch="*",
    )
    s3.get_object.assert_called_once_with(Bucket="bucket", Key="key")


@pytest.mark.parametrize("is_pr", [False, True])
@pytest.mark.parametrize("variation", ["identical", "timestamps", "member_order"])
def test_identical_content_retry_succeeds_without_overwrite(is_pr, variation):
    stored = wheel()
    fresh = wheel(year=2021) if variation == "timestamps" else stored
    if variation == "member_order":
        fresh = wheel([("metadata", b"version"), ("package.py", b"code")])
    if variation != "identical":
        assert fresh != stored
    s3 = client(stored, "PreconditionFailed")
    assert publish_wheel(s3, "bucket", "key", fresh, is_pr) == hashlib.sha256(stored).hexdigest()
    s3.put_object.assert_called_once_with(
        Bucket="bucket",
        Key="key",
        Body=fresh,
        IfNoneMatch="*",
    )
    assert s3.get_object.return_value["Body"].closed


@pytest.mark.parametrize("is_pr", [False, True])
@pytest.mark.parametrize("error", [None, "PreconditionFailed"])
@pytest.mark.parametrize(
    "members",
    [
        [("package.py", b"changed"), ("metadata", b"version")],
        [("renamed.py", b"code"), ("metadata", b"version")],
        [("package.py", b"code")],
        [("package.py", b"code"), ("metadata", b"version"), ("extra", b"")],
    ],
)
def test_different_content_fails(is_pr, error, members):
    s3 = client(wheel(members), error)
    with pytest.raises(RuntimeError, match="Stored wheel differs.*refusing to overwrite") as exc:
        publish_wheel(s3, "bucket", "key", wheel(), is_pr)
    assert ("Bump __version__" in str(exc.value)) == (not is_pr)
    assert s3.put_object.call_count == 1
    assert s3.get_object.return_value["Body"].closed


@pytest.mark.parametrize("is_pr", [False, True])
@pytest.mark.parametrize("error", ["AccessDenied", "ConditionalRequestConflict"])
def test_other_s3_errors_are_reraised(is_pr, error):
    s3 = client(error=error)
    with pytest.raises(ClientError) as exc:
        publish_wheel(s3, "bucket", "key", wheel(), is_pr)
    assert exc.value is s3.put_object.side_effect
    s3.get_object.assert_not_called()


def test_unreadable_existing_artifact_fails_closed():
    s3 = client(error="PreconditionFailed")
    s3.get_object.side_effect = ClientError({"Error": {"Code": "AccessDenied"}}, "GetObject")
    with pytest.raises(ClientError, match="AccessDenied"):
        publish_wheel(s3, "bucket", "key", wheel(), True)


@pytest.mark.parametrize("is_pr", [False, True])
def test_invalid_existing_archive_fails_closed(is_pr):
    s3 = client(b"not a zip", "PreconditionFailed")
    with pytest.raises(RuntimeError, match="Stored wheel differs"):
        publish_wheel(s3, "bucket", "key", wheel(), is_pr)


@pytest.mark.parametrize("is_pr", [False, True])
def test_receipt_records_stored_digest(tmp_path, monkeypatch, is_pr):
    stored, fresh = wheel(), wheel(year=2021)
    s3 = client(stored, "PreconditionFailed")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("WHEEL", "test.whl")
    monkeypatch.setenv("KEY", "test-key")
    monkeypatch.setenv("ALLOW_IDENTICAL_PR_ARTIFACT", str(is_pr).lower())
    monkeypatch.setattr(publisher["boto3"], "client", lambda service: s3)
    Path("dist").mkdir()
    Path("dist/test.whl").write_bytes(fresh)
    Path("published.sha256").write_text("stale receipt")
    publisher["main"]()
    assert Path("published.sha256").read_text() == hashlib.sha256(stored).hexdigest() + "  test.whl\n"


def test_failed_retry_removes_stale_receipt(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("WHEEL", "test.whl")
    monkeypatch.setenv("KEY", "test-key")
    monkeypatch.setattr(
        publisher["boto3"],
        "client",
        lambda service: client(b"bad", "PreconditionFailed"),
    )
    Path("dist").mkdir()
    Path("dist/test.whl").write_bytes(wheel())
    Path("published.sha256").write_text("stale receipt")
    with pytest.raises(RuntimeError):
        publisher["main"]()
    assert not Path("published.sha256").exists()
