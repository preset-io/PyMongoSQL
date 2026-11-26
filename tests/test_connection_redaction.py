# -*- coding: utf-8 -*-
"""Connection logs, reprs and errors must never carry the MongoDB password."""

import logging
import traceback
from unittest import mock
from urllib.parse import quote_plus

import pytest
from pymongo.errors import ServerSelectionTimeoutError

from pymongosql.connection import Connection
from pymongosql.error import DatabaseError, OperationalError
from pymongosql.helper import ConnectionHelper
from tests.conftest import TEST_URI

# Placeholder credentials; the special-character one needs percent-encoding in a URI.
PLAIN_PASSWORD = "Plain-Placeholder-Pw-1"
SPECIAL_PASSWORD = "Sp3c!al@Pw:/?#%+&="
CERT_PASSWORD = "Cert-Placeholder-Pw-2"

URIS = [
    pytest.param(f"mongodb://appuser:{PLAIN_PASSWORD}@db.example.net:27017/sales", PLAIN_PASSWORD, id="mongodb"),
    pytest.param(
        f"mongodb+srv://appuser:{PLAIN_PASSWORD}@cluster0.example.net/sales?retryWrites=true&mode=superset",
        PLAIN_PASSWORD,
        id="mongodb+srv",
    ),
    pytest.param(
        f"mongodb://appuser:{quote_plus(SPECIAL_PASSWORD)}@h1.example.net:27017,h2.example.net:27018/sales"
        "?replicaSet=rs0&authSource=admin",
        SPECIAL_PASSWORD,
        id="special-chars-replica-set",
    ),
    pytest.param(
        f"mongodb+srv://appuser:{quote_plus(SPECIAL_PASSWORD)}@cluster0.example.net/sales"
        f"?tls=true&tlsCertificateKeyFilePassword={CERT_PASSWORD}",
        SPECIAL_PASSWORD,
        id="srv-special-chars-cert-password",
    ),
]


def _forms(password):
    """The raw, percent-encoded and partly encoded forms a password can appear in."""
    return {password, quote_plus(password), password.replace("@", "%40")}


def _assert_clean(text, *passwords):
    for password in passwords:
        for form in _forms(password):
            assert form not in text, f"password leaked: {text!r}"


def _traceback_text(exc):
    return "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))


@pytest.fixture
def fake_client():
    """Replace MongoClient so connecting succeeds without a server or DNS lookup."""
    with mock.patch("pymongosql.connection.MongoClient") as client_cls:
        client_cls.return_value.nodes = frozenset()
        yield client_cls


@pytest.mark.parametrize("uri, password", URIS)
def test_success_log_has_no_password(caplog, fake_client, uri, password):
    with caplog.at_level(logging.DEBUG, logger="pymongosql"):
        conn = Connection(host=uri)

    assert "Successfully connected to MongoDB at" in caplog.text
    assert "example.net" in caplog.text
    _assert_clean(caplog.text, password, CERT_PASSWORD)
    _assert_clean(str(conn), password, CERT_PASSWORD)
    conn.close()
    _assert_clean(conn.host, password, CERT_PASSWORD)
    _assert_clean(caplog.text, password, CERT_PASSWORD)


def test_success_log_names_host_and_database(caplog, fake_client):
    uri = f"mongodb+srv://appuser:{PLAIN_PASSWORD}@cluster0.example.net/sales?retryWrites=true"
    with caplog.at_level(logging.INFO, logger="pymongosql"):
        Connection(host=uri)

    assert "Successfully connected to MongoDB at mongodb+srv://cluster0.example.net/sales (database: sales)" in (
        caplog.text
    )


def test_password_keyword_argument_not_logged(caplog, fake_client):
    with caplog.at_level(logging.DEBUG, logger="pymongosql"):
        conn = Connection(host="db.example.net", username="appuser", password=PLAIN_PASSWORD, database="sales")

    assert "Successfully connected to MongoDB at db.example.net:27017" in caplog.text
    _assert_clean(caplog.text + str(conn), PLAIN_PASSWORD)


@pytest.mark.parametrize("uri, password", URIS)
def test_connection_failure_message_has_no_password(caplog, fake_client, uri, password):
    # A driver error that echoes the URI must be scrubbed from the log, the raised
    # error and the traceback (no chained original exception).
    fake_client.return_value.admin.command.side_effect = ServerSelectionTimeoutError(f"cannot reach {uri}")
    with caplog.at_level(logging.DEBUG, logger="pymongosql"):
        with pytest.raises(OperationalError) as exc_info:
            Connection(host=uri)

    assert "Failed to connect to MongoDB" in caplog.text
    _assert_clean(caplog.text, password, CERT_PASSWORD)
    _assert_clean(_traceback_text(exc_info.value), password, CERT_PASSWORD)


def test_unexpected_error_message_has_no_password(caplog, fake_client):
    fake_client.side_effect = ValueError(f"bad option, password={PLAIN_PASSWORD}")
    with caplog.at_level(logging.DEBUG, logger="pymongosql"):
        with pytest.raises(DatabaseError) as exc_info:
            Connection(host="db.example.net", username="appuser", password=PLAIN_PASSWORD, database="sales")

    _assert_clean(caplog.text, PLAIN_PASSWORD)
    _assert_clean(_traceback_text(exc_info.value), PLAIN_PASSWORD)


@pytest.mark.parametrize("connect", [True, False])
@pytest.mark.parametrize("lead", ["LeadPart", "13579"])
def test_unescaped_slash_in_password_leaks_no_fragment(caplog, connect, lead):
    # urlparse and pymongo both split an unescaped "/" in the password into the path.
    # pymongo then rejects the non-digit lead part as a port, or takes a digit one as
    # the port of host "appuser" and echoes it in its timeout error. The database name
    # parsed from the URI must not be the rest of the password either.
    uri = f"mongodb://appuser:{lead}/TailSecretPart@db.example.net/sales"
    with caplog.at_level(logging.DEBUG, logger="pymongosql"):
        try:
            # A digit lead part makes a URI pymongo accepts (host "appuser", port <lead>).
            conn = Connection(host=uri, connect=connect, serverSelectionTimeoutMS=100)
            text = str(conn) + conn.host
            conn.close()
        except Exception as e:
            text = _traceback_text(e)

    text += caplog.text
    assert "TailSecretPart" not in text
    assert lead not in text
    assert ConnectionHelper.parse_connection_string(uri)[1] == "sales"


def test_clean_errors_keep_their_original_type():
    # Scrubbing must not change errors that carry no secret.
    with pytest.raises(Exception) as exc_info:
        Connection(host="mongodb://db.example.net/sales?notAnOption", connect=False)
    assert not isinstance(exc_info.value, DatabaseError)


@pytest.mark.parametrize(
    "text, expected",
    [
        ("mongodb://u:pw@h/db", "mongodb://u:***@h/db"),
        ("at mongodb+srv://u:p@ss@h.example.net/db?x=1 now", "at mongodb+srv://u:***@h.example.net/db?x=1 now"),
        ("mongodb://h:27017/db", "mongodb://h:27017/db"),
        ("mongodb://u@h/db", "mongodb://u@h/db"),
    ],
)
def test_redact_uri(text, expected):
    assert ConnectionHelper.redact_uri(text) == expected


@pytest.mark.parametrize(
    "host, expected",
    [
        (None, "localhost:27017"),
        ("localhost", "localhost:27017"),
        ("h1:27018", "h1:27018"),
        ("mongodb://u:pw@h1:1,h2:2/db?authSource=admin", "mongodb://h1:1,h2:2/db"),
        (
            "mongodb+srv://u:pw@cluster0.example.net/?tlsCertificateKeyFilePassword=x",
            "mongodb+srv://cluster0.example.net/",
        ),
        (["mongodb://u:pw@h1/db", "h2"], "mongodb://h1/db,h2:27017"),
    ],
)
def test_describe_host(host, expected):
    assert ConnectionHelper.describe_host(host, 27017) == expected


@pytest.mark.skipif(bool(TEST_URI), reason="uses the local test server credentials")
def test_real_connection_log_has_no_password(caplog):
    """Against the test server: the URI with credentials must not be logged."""
    with caplog.at_level(logging.DEBUG, logger="pymongosql"):
        conn = Connection(host="mongodb://testuser:testpass@localhost:27017/test_db?authSource=test_db")
        conn.close()

    assert "Successfully connected to MongoDB at mongodb://localhost:27017/test_db" in caplog.text
    assert "testpass" not in caplog.text
