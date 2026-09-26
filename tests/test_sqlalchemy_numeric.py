# -*- coding: utf-8 -*-
"""Decimal values must round-trip exactly through the SQLAlchemy dialect."""

from decimal import Decimal

import pytest

from pymongosql.helper import SQLHelper
from tests.conftest import HAS_SQLALCHEMY

pytestmark = pytest.mark.skipif(not HAS_SQLALCHEMY, reason="SQLAlchemy not available")

if HAS_SQLALCHEMY:
    import sqlalchemy as sa
    from bson import Decimal128

    from pymongosql.sqlalchemy_mongodb.sqlalchemy_dialect import PyMongoSQLDialect

EXACT = Decimal("123456789012345678901.1234567890")
TINY = Decimal("-0.0000000001")


def result_processor(type_):
    dialect = PyMongoSQLDialect()
    return type_.dialect_impl(dialect).result_processor(dialect, None) or (lambda v: v)


class TestOffline:
    def test_decimal_parameters_are_encoded_as_decimal128(self):
        replaced = SQLHelper.replace_placeholders_generic({"a": "?", "b": {"$in": ["?"]}}, [EXACT, TINY], "qmark")
        assert replaced == {"a": Decimal128(EXACT), "b": {"$in": [Decimal128(TINY)]}}

    def test_named_decimal_parameters_are_encoded_as_decimal128(self):
        replaced = SQLHelper.replace_placeholders_generic({"a": ":v"}, {"v": EXACT}, "named")
        assert replaced == {"a": Decimal128(EXACT)}

    def test_other_parameters_are_unchanged(self):
        values = [1, 1.5, "s", None, True]
        replaced = SQLHelper.replace_placeholders_generic(["?"] * 5, values, "qmark")
        assert replaced == values

    def test_numeric_column_returns_decimal(self):
        value = result_processor(sa.Numeric(31, 10))(Decimal128(EXACT))
        assert value == EXACT and type(value) is Decimal

    def test_float_column_returns_float(self):
        value = result_processor(sa.Float())(Decimal128("1.25"))
        assert value == 1.25 and type(value) is float

    def test_integer_columns_return_int_for_int64(self):
        from bson import Int64

        for type_ in (sa.Integer(), sa.BigInteger()):
            value = result_processor(type_)(Int64(2**62))
            assert value == 2**62 and type(value) is int

    def test_numeric_column_passes_other_values_through(self):
        processor = result_processor(sa.Numeric(31, 10))
        assert processor(None) is None


class TestLive:
    def test_decimal_roundtrip(self, sqlalchemy_engine, conn):
        table = sa.Table(
            "test_decimal_roundtrip",
            sa.MetaData(),
            sa.Column("id", sa.Integer),
            sa.Column("amount", sa.Numeric(31, 10)),
        )
        conn.database.drop_collection(table.name)
        try:
            with sqlalchemy_engine.begin() as connection:
                connection.execute(table.insert(), [{"id": 1, "amount": EXACT}, {"id": 2, "amount": TINY}])
            stored = [d["amount"] for d in conn.database[table.name].find({}, sort=[("id", 1)])]
            assert stored == [Decimal128(EXACT), Decimal128(TINY)]
            with sqlalchemy_engine.connect() as connection:
                amounts = connection.execute(
                    sa.select(sa.column("amount", sa.Numeric(31, 10))).select_from(sa.table(table.name))
                ).scalars()
                values = sorted(amounts)
            assert values == [TINY, EXACT]
            assert all(type(v) is Decimal for v in values)
        finally:
            conn.database.drop_collection(table.name)

    def test_int64_roundtrip(self, sqlalchemy_engine, conn):
        table = sa.Table("test_int64_roundtrip", sa.MetaData(), sa.Column("big", sa.BigInteger))
        conn.database.drop_collection(table.name)
        try:
            conn.database[table.name].insert_many([{"big": 2**63 - 1}, {"big": -(2**63)}])
            with sqlalchemy_engine.connect() as connection:
                values = sorted(connection.execute(sa.select(table.c.big)).scalars())
            assert values == [-(2**63), 2**63 - 1]
            assert all(type(v) is int for v in values)
        finally:
            conn.database.drop_collection(table.name)
