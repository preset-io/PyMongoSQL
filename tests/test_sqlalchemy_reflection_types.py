# -*- coding: utf-8 -*-
"""Offline tests for column type inference in SQLAlchemy reflection."""

import uuid
from datetime import datetime
from unittest.mock import MagicMock

import pytest

sqlalchemy = pytest.importorskip("sqlalchemy")

from bson import Binary, Decimal128, Int64, ObjectId  # noqa: E402
from sqlalchemy import types  # noqa: E402
from sqlalchemy.sql.sqltypes import NullType  # noqa: E402

from pymongosql.sqlalchemy_mongodb.sqlalchemy_dialect import PyMongoSQLDialect  # noqa: E402


def reflect(documents):
    """Run get_columns against an in-memory sample instead of a server."""
    collection = MagicMock()
    collection.find.return_value.limit.return_value = documents
    database = MagicMock()
    database.__getitem__.return_value = collection
    client = MagicMock()
    client.__getitem__.return_value = database
    connection = MagicMock()
    connection.connection._client = client
    columns = PyMongoSQLDialect().get_columns(connection, "c", schema="db")
    return {c["name"]: c["type"] for c in columns}


def is_type(reflected, expected):
    reflected = reflected if isinstance(reflected, type) else type(reflected)
    return issubclass(reflected, expected)


def test_decimal128_reflects_as_numeric():
    reflected = reflect([{"_id": 1, "amount": Decimal128("123456789012345678901.1234567890")}])
    assert is_type(reflected["amount"], types.Numeric)
    assert not is_type(reflected["amount"], types.Float)


def test_leading_null_uses_the_first_non_null_value():
    reflected = reflect([{"_id": 1, "optional": None}, {"_id": 2, "optional": "present"}])
    assert is_type(reflected["optional"], types.String)


def test_all_null_field_stays_null_type():
    reflected = reflect([{"_id": 1, "optional": None}, {"_id": 2, "optional": None}])
    assert is_type(reflected["optional"], NullType)


def test_first_non_null_type_is_not_overwritten_by_later_values():
    reflected = reflect([{"_id": 1, "n": 1}, {"_id": 2, "n": "text"}])
    assert is_type(reflected["n"], types.Integer)


@pytest.mark.parametrize(
    "value,expected",
    [
        (ObjectId(), types.String),
        (Int64(2**40), types.BigInteger),
        (Binary(b"\x00\x01"), types.LargeBinary),
        (b"\x00\x01", types.LargeBinary),
        (uuid.uuid4(), types.String),
        (datetime(2026, 1, 1), types.DateTime),
        (True, types.Boolean),
        (1.5, types.Float),
    ],
)
def test_bson_values_map_to_their_sqlalchemy_types(value, expected):
    assert is_type(reflect([{"_id": 1, "v": value}])["v"], expected)
