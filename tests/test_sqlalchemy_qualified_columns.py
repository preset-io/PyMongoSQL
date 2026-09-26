# -*- coding: utf-8 -*-
"""Table-qualified column references must read the column, not a nested path."""

import pytest

from pymongosql.sql.parser import SQLParser
from tests.conftest import HAS_SQLALCHEMY

if HAS_SQLALCHEMY:
    import sqlalchemy as sa

    from pymongosql.sqlalchemy_mongodb.sqlalchemy_dialect import PyMongoSQLDialect

    TABLE = sa.Table(
        "users",
        sa.MetaData(),
        sa.Column("_id", sa.String, primary_key=True),
        sa.Column("name", sa.String),
        sa.Column("age", sa.Integer),
    )

needs_sqlalchemy = pytest.mark.skipif(not HAS_SQLALCHEMY, reason="SQLAlchemy not available")


def plan(sql):
    return SQLParser(sql).get_execution_plan()


class TestParser:
    def test_qualified_projection_filter_and_sort(self):
        p = plan("SELECT users.name, users.age FROM users WHERE users.age > 30 ORDER BY users.age DESC")
        assert p.projection_stage == {"name": 1, "age": 1}
        assert p.filter_stage == {"age": {"$gt": 30}}
        assert p.sort_stage == [{"age": -1}]

    def test_qualified_nested_path_keeps_the_path(self):
        assert plan("SELECT users.profile.bio FROM users").projection_stage == {"profile.bio": 1}

    def test_other_prefix_is_still_a_nested_path(self):
        assert plan("SELECT profile.bio FROM users").projection_stage == {"profile.bio": 1}

    def test_qualified_aggregate_argument(self):
        p = plan("SELECT SUM(users.age) AS total FROM users")
        assert '"$sum": "$age"' in p.aggregate_pipeline


@needs_sqlalchemy
class TestCompiler:
    def test_core_select_renders_unqualified_columns(self):
        stmt = sa.select(TABLE.c.name, TABLE.c.age).where(TABLE.c._id == "x").order_by(TABLE.c.age)
        sql = " ".join(str(stmt.compile(dialect=PyMongoSQLDialect())).split())
        assert sql == "SELECT name, age FROM users WHERE _id = ? ORDER BY age"


@needs_sqlalchemy
class TestLive:
    def test_core_select_returns_values(self, sqlalchemy_engine, conn):
        expected = conn.database["users"].find_one({"_id": "1"}, {"name": 1, "age": 1})
        with sqlalchemy_engine.connect() as connection:
            row = connection.execute(sa.select(TABLE.c.name, TABLE.c.age).where(TABLE.c._id == "1")).one()
            full = connection.execute(sa.select(TABLE).where(TABLE.c._id == "1")).mappings().one()
        assert tuple(row) == (expected["name"], expected["age"])
        assert (full["name"], full["age"]) == (expected["name"], expected["age"])

    def test_raw_qualified_sql_returns_values(self, conn):
        expected = conn.database["users"].find_one({"_id": "1"}, {"name": 1})["name"]
        cursor = conn.cursor()
        cursor.execute("SELECT users.name FROM users WHERE users._id = '1'")
        assert cursor.fetchall() == [(expected,)]
