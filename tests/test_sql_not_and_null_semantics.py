# -*- coding: utf-8 -*-
"""NOT and NULL follow SQL three-valued logic; an untranslatable WHERE never widens a query."""

import pytest

from pymongosql.error import Error
from pymongosql.sql.parser import SQLParser
from tests.conftest import HAS_SQLALCHEMY

COLLECTION = "test_not_semantics"
# _id 4 has a NULL a; _id 5 has no a at all. SQL never returns them for a
# comparison on a, negated or not.
DOCS = [
    {"_id": 1, "a": 1, "b": "x", "flag": True},
    {"_id": 2, "a": 2, "b": "y", "flag": False},
    {"_id": 3, "a": 3, "b": "xz", "flag": True},
    {"_id": 4, "a": None, "b": None, "flag": None},
    {"_id": 5},
]


def plan(sql):
    return SQLParser(sql).get_execution_plan()


class TestPlans:
    def test_not_comparison_excludes_null(self):
        assert plan("SELECT _id FROM t WHERE NOT a = 1").filter_stage == {"a": {"$nin": [1, None]}}

    def test_not_or_uses_de_morgan(self):
        assert plan("SELECT _id FROM t WHERE NOT (a = 1 OR b = 'x')").filter_stage == {
            "$and": [{"a": {"$nin": [1, None]}}, {"b": {"$nin": ["x", None]}}]
        }

    def test_double_not(self):
        assert plan("SELECT _id FROM t WHERE NOT NOT a = 1").filter_stage == {"a": 1}

    def test_not_between_and_not_is_null(self):
        assert plan("SELECT _id FROM t WHERE NOT a BETWEEN 1 AND 2").filter_stage == {
            "$or": [{"a": {"$lt": 1}}, {"a": {"$gt": 2}}]
        }
        assert plan("SELECT _id FROM t WHERE NOT a IS NULL").filter_stage == {"a": {"$ne": None}}

    def test_not_bare_boolean_field(self):
        assert plan("SELECT _id FROM t WHERE NOT flag").filter_stage == {"flag": False}

    def test_not_in_list_containing_null_is_never_true(self):
        assert plan("SELECT _id FROM t WHERE a NOT IN (1, NULL)").filter_stage == {"$expr": False}

    def test_not_equal_excludes_null(self):
        assert plan("SELECT _id FROM t WHERE a <> 1").filter_stage == {"a": {"$nin": [1, None]}}

    @pytest.mark.parametrize(
        "sql",
        [
            "SELECT _id FROM t WHERE NOT lower(b) = 'x'",
            "SELECT _id FROM t WHERE lower(b) = 'x'",
            "SELECT _id FROM t WHERE a = 1 AND lower(b) = 'x'",
            "SELECT _id FROM t WHERE b LIKE ?",
        ],
    )
    def test_untranslatable_where_raises_instead_of_widening(self, sql):
        with pytest.raises(Error):
            plan(sql)

    @pytest.mark.parametrize("sql", ["DELETE FROM t WHERE lower(b) = 'x'", "UPDATE t SET a = 1 WHERE lower(b) = 'x'"])
    def test_untranslatable_dml_where_raises_instead_of_matching_everything(self, sql):
        with pytest.raises(Error):
            plan(sql)

    def test_not_in_delete(self):
        assert plan("DELETE FROM t WHERE NOT a = 1").filter_conditions == {"a": {"$nin": [1, None]}}


@pytest.fixture
def docs(conn):
    conn.database.drop_collection(COLLECTION)
    conn.database[COLLECTION].insert_many(DOCS)
    yield conn
    conn.database.drop_collection(COLLECTION)


def ids(conn, where, params=None):
    cursor = conn.cursor()
    sql = f"SELECT _id FROM {COLLECTION} WHERE {where} ORDER BY _id"
    cursor.execute(sql, params) if params is not None else cursor.execute(sql)
    return [r[0] for r in cursor.fetchall()]


class TestLive:
    @pytest.mark.parametrize(
        "where,expected",
        [
            ("NOT a = 1", [2, 3]),
            ("NOT (a = 1 OR b = 'y')", [3]),
            ("NOT (a = 1 AND b = 'x')", [2, 3]),
            ("NOT a IN (1, 2)", [3]),
            ("NOT a NOT IN (1, 2)", [1, 2]),
            ("NOT b LIKE 'x%'", [2]),
            ("NOT a BETWEEN 2 AND 3", [1]),
            ("NOT a IS NULL", [1, 2, 3]),
            ("NOT flag", [2]),
            ("a <> 1", [2, 3]),
            ("b NOT LIKE 'x%'", [2]),
            ("a NOT IN (1)", [2, 3]),
            ("a = 1 OR NOT (b = 'x' OR b = 'xz')", [1, 2]),
        ],
    )
    def test_negation_returns_sql_rows(self, docs, where, expected):
        assert ids(docs, where) == expected

    def test_not_with_bound_parameter(self, docs):
        assert ids(docs, "NOT a = ?", [2]) == [1, 3]

    def test_untranslatable_delete_deletes_nothing(self, docs):
        cursor = docs.cursor()
        with pytest.raises(Error):
            cursor.execute(f"DELETE FROM {COLLECTION} WHERE lower(b) = 'x'")
        assert docs.database[COLLECTION].count_documents({}) == len(DOCS)


@pytest.mark.skipif(not HAS_SQLALCHEMY, reason="SQLAlchemy not available")
class TestSQLAlchemy:
    def test_like_pattern_is_rendered_inline(self):
        import sqlalchemy as sa

        from pymongosql.sqlalchemy_mongodb.sqlalchemy_dialect import PyMongoSQLDialect

        t = sa.table("t", sa.column("b"))
        stmt = sa.select(t.c.b).where(t.c.b.like("O'B%"), t.c.b.not_like("x_"))
        sql = " ".join(str(stmt.compile(dialect=PyMongoSQLDialect())).split())
        assert sql == "SELECT b FROM t WHERE b LIKE 'O''B%' AND b NOT LIKE 'x_'"

    def test_core_not_and_like_rows(self, sqlalchemy_engine, docs):
        import sqlalchemy as sa

        t = sa.table(COLLECTION, sa.column("_id"), sa.column("a"), sa.column("b"))
        with sqlalchemy_engine.connect() as connection:
            got = connection.execute(
                sa.select(t.c._id).where(sa.not_(t.c.a == 1), t.c.b.like("x%")).order_by(t.c._id)
            ).scalars()
            assert list(got) == [3]
