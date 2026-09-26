# -*- coding: utf-8 -*-
"""FROM aliases resolve to the collection; untranslatable FROM clauses raise."""

import pytest

from pymongosql.error import Error
from pymongosql.sql.parser import SQLParser
from tests.conftest import HAS_SQLALCHEMY

COLLECTION = "test_from_alias"
DOCS = [
    {"_id": 1, "g": "a", "v": 10, "profile": {"city": "x"}},
    {"_id": 2, "g": "a", "v": 20, "profile": {"city": "y"}},
    {"_id": 3, "g": "b", "v": 35, "profile": {"city": "x"}},
]


def plan(sql):
    return SQLParser(sql).get_execution_plan()


class TestPlans:
    @pytest.mark.parametrize("from_clause", ["t AS x", "t x", 't AS "x"'])
    def test_alias_qualified_columns_resolve_to_fields(self, from_clause):
        p = plan(f"SELECT x.a, x.b AS bee FROM {from_clause} WHERE x.b = 1 AND NOT x.c = 2 ORDER BY x.a")
        assert p.collection == "t"
        assert p.projection_stage == {"a": 1, "b": 1}
        assert p.column_aliases == {"b": "bee"}
        assert p.filter_stage == {"$and": [{"b": 1}, {"c": {"$nin": [2, None]}}]}
        assert p.sort_stage == [{"a": 1}]

    def test_alias_with_nested_path(self):
        assert plan("SELECT x.profile.city FROM t AS x").projection_stage == {"profile.city": 1}

    def test_quoted_collection_with_alias(self):
        p = plan('SELECT ua.a FROM "user.accounts" AS ua')
        assert (p.collection, p.projection_stage) == ("user.accounts", {"a": 1})

    def test_qualified_group_by_keys(self):
        for sql in (
            "SELECT t.g, SUM(t.v) AS s FROM t GROUP BY t.g",
            "SELECT x.g, SUM(x.v) AS s FROM t x GROUP BY x.g",
        ):
            assert '"_id": {"g0": "$g"}' in plan(sql).aggregate_pipeline
            assert '"$sum": "$v"' in plan(sql).aggregate_pipeline

    @pytest.mark.parametrize(
        "sql",
        [
            "SELECT a FROM t, u",
            "SELECT a FROM t JOIN u ON t.a = u.a",
            "SELECT a FROM (SELECT a FROM t) AS v",
            "SELECT a FROM t AS x AT i",
        ],
    )
    def test_untranslatable_from_raises(self, sql):
        with pytest.raises(Error, match="Unsupported SQL clause: FROM"):
            plan(sql)


@pytest.fixture
def docs(conn):
    conn.database.drop_collection(COLLECTION)
    conn.database[COLLECTION].insert_many(DOCS)
    yield conn
    conn.database.drop_collection(COLLECTION)


def rows(conn, sql):
    cursor = conn.cursor()
    cursor.execute(sql)
    return [tuple(r) for r in cursor.fetchall()], [d[0] for d in cursor.description]


class TestLive:
    def test_aliased_select_returns_rows(self, docs):
        got, names = rows(
            docs, f"SELECT x._id, x.profile.city AS city FROM {COLLECTION} AS x WHERE x.v > 10 ORDER BY x._id"
        )
        assert got == [(2, "y"), (3, "x")]
        assert names == ["_id", "city"]

    def test_aliased_group_by_returns_rows(self, docs):
        got, _ = rows(docs, f"SELECT x.g, SUM(x.v) AS s FROM {COLLECTION} x GROUP BY x.g ORDER BY s DESC")
        assert got == [("b", 35), ("a", 30)]
        got, _ = rows(docs, f"SELECT x.g, COUNT(*) AS n FROM {COLLECTION} x GROUP BY x.g ORDER BY x.g")
        assert got == [("a", 2), ("b", 1)]


@pytest.mark.skipif(not HAS_SQLALCHEMY, reason="SQLAlchemy not available")
class TestSQLAlchemy:
    def test_core_alias(self, sqlalchemy_engine, docs):
        import sqlalchemy as sa

        t = sa.table(COLLECTION, sa.column("_id"), sa.column("v")).alias("x")
        with sqlalchemy_engine.connect() as connection:
            got = connection.execute(sa.select(t.c._id).where(t.c.v >= 20).order_by(t.c._id)).scalars()
            assert list(got) == [2, 3]
