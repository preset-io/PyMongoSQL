"""Publish an immutable wheel; retries may reuse identical archive content."""

import hashlib
import io
import os
from pathlib import Path
from zipfile import BadZipFile, ZipFile

import boto3
from botocore.exceptions import ClientError


def same_wheel_content(stored, fresh):
    """Compare sorted member names and bytes, ignoring ZIP metadata such as timestamps."""
    if stored == fresh:
        return True
    try:
        with ZipFile(io.BytesIO(stored)) as old, ZipFile(io.BytesIO(fresh)) as new:
            old_members = sorted(old.infolist(), key=lambda member: member.filename)
            new_members = sorted(new.infolist(), key=lambda member: member.filename)
            return [member.filename for member in old_members] == [member.filename for member in new_members] and all(
                old.read(a) == new.read(b) for a, b in zip(old_members, new_members)
            )
    except BadZipFile:
        return False


def publish_wheel(s3, bucket, key, body, is_pr=False):
    try:
        s3.put_object(Bucket=bucket, Key=key, Body=body, IfNoneMatch="*")
    except ClientError as error:
        if error.response["Error"]["Code"] != "PreconditionFailed":
            raise
        print("Artifact already exists; verifying archive content without overwriting.")

    response = s3.get_object(Bucket=bucket, Key=key)
    try:
        stored = response["Body"].read()
    finally:
        response["Body"].close()
    if not same_wheel_content(stored, body):
        raise RuntimeError(
            "Stored wheel differs from this commit's freshly built artifact; "
            "refusing to overwrite or accept it (local sha256={}, stored sha256={}).".format(
                hashlib.sha256(body).hexdigest(),
                hashlib.sha256(stored).hexdigest(),
            )
            + ("" if is_pr else " Bump __version__ before publishing different content.")
        )
    print("Published wheel verified by archive content: " + key)
    return hashlib.sha256(stored).hexdigest()


def main():
    wheel = os.environ["WHEEL"]
    receipt = Path("published.sha256")
    # Do not leave a previous build's success receipt after a failed retry.
    if receipt.exists():
        receipt.unlink()
    digest = publish_wheel(
        boto3.client("s3"),
        "preset-pypi",
        os.environ["KEY"],
        Path("dist", wheel).read_bytes(),
        is_pr=os.environ.get("ALLOW_IDENTICAL_PR_ARTIFACT") == "true",
    )
    receipt.write_text(digest + "  " + wheel + "\n")


if __name__ == "__main__":
    main()
