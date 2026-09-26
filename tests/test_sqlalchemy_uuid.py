# -*- coding: utf-8 -*-
"""SQLAlchemy 2 Uuid columns must round-trip BSON UUIDs."""

import uuid

import pytest

from tests.conftest import HAS_SQLALCHEMY

sa = pytest.importorskip("sqlalchemy")
pytestmark = pytest.mark.skipif(not (HAS_SQLALCHEMY and hasattr(sa, "Uuid")), reason="needs SQLAlchemy 2 Uuid")

from bson.binary import Binary, UuidRepresentation  # noqa: E402

from pymongosql.sqlalchemy_mongodb.sqlalchemy_dialect import PyMongoSQLDialect  # noqa: E402

VALUE = uuid.UUID("00000000-0000-4000-8000-000000000001")


def processors(type_):
    dialect = PyMongoSQLDialect()
    impl = type_.dialect_impl(dialect)
    return impl.bind_processor(dialect), impl.result_processor(dialect, None)


@pytest.mark.parametrize("stored", [VALUE, Binary.from_uuid(VALUE), str(VALUE)])
def test_uuid_column_reads_uuid(stored):
    _, result = processors(sa.Uuid())
    assert result(stored) == VALUE


def test_uuid_column_as_string():
    _, result = processors(sa.Uuid(as_uuid=False))
    assert result(VALUE) == str(VALUE)


def test_uuid_bind_is_standard_binary():
    bind, _ = processors(sa.Uuid())
    assert bind(VALUE) == Binary.from_uuid(VALUE)
    assert bind(str(VALUE)) == Binary.from_uuid(VALUE)
    assert bind(None) is None


def test_legacy_subtype_3_is_not_guessed():
    legacy = Binary.from_uuid(VALUE, UuidRepresentation.PYTHON_LEGACY)
    _, result = processors(sa.Uuid())
    assert result(legacy) == legacy


def test_live_roundtrip(sqlalchemy_engine, conn):
    table = sa.Table("test_uuid_roundtrip", sa.MetaData(), sa.Column("id", sa.Integer), sa.Column("u", sa.Uuid))
    conn.database.drop_collection(table.name)
    try:
        with sqlalchemy_engine.begin() as connection:
            connection.execute(table.insert(), [{"id": 1, "u": VALUE}])
        assert conn.database[table.name].find_one()["u"] == Binary.from_uuid(VALUE)
        with sqlalchemy_engine.connect() as connection:
            assert connection.execute(sa.select(table.c.u)).scalar_one() == VALUE
    finally:
        conn.database.drop_collection(table.name)
