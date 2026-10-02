# -*- coding: utf-8 -*-
"""Quoted operator names and LIKE ... ESCAPE precedence.

- A field name starting with ``$`` is rejected whether or not it is quoted, so
  ``"$where"`` or ``"$expr"`` can never reach MongoDB as a query operator.
- ``LIKE ... ESCAPE`` keeps SQL precedence for the conditions that follow it: the
  grammar parses the rest of the boolean chain as the escape expression.
"""

import pytest

from pymongosql.error import Error
from pymongosql.sql.parser import SQLParser
from tests.conftest import HAS_SQLALCHEMY

if HAS_SQLALCHEMY:
    import sqlalchemy as sa

needs_sqlalchemy = pytest.mark.skipif(not HAS_SQLALCHEMY, reason="SQLAlchemy not available")


def plan(sql):
    return SQLParser(sql).get_execution_plan()


def where(sql_where):
    return plan(f"SELECT _id FROM t WHERE {sql_where}").filter_stage


def run(conn, sql, params=None):
    cursor = conn.cursor()
    cursor.execute(sql, params) if params is not None else cursor.execute(sql)
    return cursor


# ------------------------------------------------------------ $-prefixed field names


class TestOperatorFieldNames:
    @pytest.mark.parametrize(
        "sql_where",
        [
            "\"$where\" = 'sleep(300) || true'",
            '"$expr" = 1',
            '"$where"',
            "NOT \"$where\" = 'true'",
            "'true' = \"$where\"",
            'n = 1 OR "$expr" = 1',
            "\"$where\" LIKE 'x%'",
            "\"$where\" IN ('true')",
            '"$where" IS NOT NULL',
            '"$expr" BETWEEN 0 AND 1',
            '"$jsonSchema" = 1',
            '"a.$where" = 1',
        ],
    )
    def test_quoted_operator_is_rejected(self, sql_where):
        with pytest.raises(Error):
            where(sql_where)

    @pytest.mark.parametrize(
        "sql",
        [
            "DELETE FROM t WHERE \"$where\" = 'true'",
            'UPDATE t SET v = 1 WHERE "$expr" = 1',
        ],
    )
    def test_quoted_operator_is_rejected_in_dml(self, sql):
        with pytest.raises(Error):
            plan(sql)

    def test_quoted_names_without_dollar_still_translate(self):
        assert where("\"first name\" = 'x'") == {"first name": "x"}
        assert where('"a.b" = 1') == {"a.b": 1}
        assert where('"price$" = 1') == {"price$": 1}


OPERATOR_COLLECTION = "test_operator_fields"


@pytest.fixture
def operator_docs(conn):
    conn.database.drop_collection(OPERATOR_COLLECTION)
    conn.database[OPERATOR_COLLECTION].insert_many([{"_id": i, "v": i} for i in (1, 2, 3)])
    yield conn
    conn.database.drop_collection(OPERATOR_COLLECTION)


class TestLiveOperatorFieldNames:
    def test_select_does_not_run_where_or_expr(self, operator_docs):
        # Before, "$expr" = 1 matched every document and "$where" ran server-side JavaScript
        with pytest.raises(Error):
            run(operator_docs, f'SELECT _id FROM {OPERATOR_COLLECTION} WHERE "$expr" = 1')
        with pytest.raises(Error):
            run(operator_docs, f"SELECT _id FROM {OPERATOR_COLLECTION} WHERE \"$where\" = 'true'")

    def test_delete_and_update_touch_nothing(self, operator_docs):
        with pytest.raises(Error):
            run(operator_docs, f'DELETE FROM {OPERATOR_COLLECTION} WHERE "$expr" = 1')
        with pytest.raises(Error):
            run(operator_docs, f"UPDATE {OPERATOR_COLLECTION} SET v = 0 WHERE \"$where\" = 'true'")
        stored = operator_docs.database[OPERATOR_COLLECTION].find({}, sort=[("_id", 1)])
        assert [(d["_id"], d["v"]) for d in stored] == [(1, 1), (2, 2), (3, 3)]


# ------------------------------------------------------------ LIKE ... ESCAPE precedence


class TestEscapePrecedence:
    def test_like_on_the_right_of_and(self):
        assert where("n = 1 AND name LIKE 'zz%' ESCAPE '/' OR n = 2") == {
            "$or": [{"$and": [{"n": 1}, {"name": {"$regex": "^zz.*"}}]}, {"n": 2}]
        }

    def test_not_applies_to_the_like_only(self):
        assert where("NOT name LIKE 'zz%' ESCAPE '/' AND n = 2") == {
            "$and": [{"$and": [{"name": {"$not": {"$regex": "^zz.*"}}}, {"name": {"$ne": None}}]}, {"n": 2}]
        }

    def test_like_first_in_the_chain_is_unchanged(self):
        assert where("n LIKE 'a/_%' ESCAPE '/' AND b = 1 OR c = 2") == {
            "$or": [{"$and": [{"n": {"$regex": "^a_.*"}}, {"b": 1}]}, {"c": 2}]
        }

    def test_escape_must_still_be_one_character(self):
        with pytest.raises(Error):
            where("n LIKE 'a' ESCAPE 'ab' AND b = 1")
        with pytest.raises(Error):
            where("n LIKE 'a' ESCAPE b = 1 AND c = 2")


ESCAPE_COLLECTION = "test_escape_precedence"
ESCAPE_DOCS = [
    {"_id": 1, "n": 1, "name": "zz_a"},
    {"_id": 2, "n": 2, "name": "abc"},
    {"_id": 3, "n": 3, "name": "zz%b"},
    {"_id": 4, "n": 4, "name": None},
    {"_id": 5, "n": 2, "name": "zzx"},
]


@pytest.fixture
def escape_docs(conn):
    conn.database.drop_collection(ESCAPE_COLLECTION)
    conn.database[ESCAPE_COLLECTION].insert_many([dict(d) for d in ESCAPE_DOCS])
    yield conn
    conn.database.drop_collection(ESCAPE_COLLECTION)


def ids(conn, sql_where, params=None):
    sql = f"SELECT _id FROM {ESCAPE_COLLECTION} WHERE {sql_where} ORDER BY _id"
    return [r[0] for r in run(conn, sql, params).fetchall()]


class TestLiveEscapePrecedence:
    @pytest.mark.parametrize(
        "sql_where,expected",
        [
            # (n = 1 AND name LIKE 'zz%') OR n = 2
            ("n = 1 AND name LIKE 'zz%' ESCAPE '/' OR n = 2", [1, 2, 5]),
            ("n = 2 AND name LIKE 'zz/%%' ESCAPE '/' OR n = 3", [3]),
            # (NOT name LIKE 'zz%') AND n = 2; a NULL name is neither LIKE nor NOT LIKE
            ("NOT name LIKE 'zz%' ESCAPE '/' AND n = 2", [2]),
            ("NOT name LIKE 'zz%' ESCAPE '/' OR n = 1", [1, 2]),
            ("n = 3 OR NOT name LIKE 'zz/_%' ESCAPE '/' AND n = 2", [2, 3, 5]),
            # Two escaped LIKEs in one chain, and a parenthesized one
            ("name LIKE 'zz/%%' ESCAPE '/' AND n = 3 OR name LIKE 'a%' ESCAPE '/' AND n = 2 OR n = 4", [2, 3, 4]),
            ("(n = 3 AND name LIKE 'zz/%%' ESCAPE '/') OR n = 2", [2, 3, 5]),
        ],
    )
    def test_rows(self, escape_docs, sql_where, expected):
        assert ids(escape_docs, sql_where) == expected

    def test_parameters_bind_in_order(self, escape_docs):
        assert ids(escape_docs, "n = ? AND name LIKE ? ESCAPE '/' OR n = ?", [1, "zz%", 2]) == [1, 2, 5]

    def test_having(self, escape_docs):
        rows = run(
            escape_docs,
            f"SELECT name, COUNT(*) AS c FROM {ESCAPE_COLLECTION} GROUP BY name "
            "HAVING COUNT(*) > 5 AND name LIKE 'zz%' ESCAPE '/' OR COUNT(*) >= 1 ORDER BY name",
        ).fetchall()
        assert [r[0] for r in rows] == [None, "abc", "zz%b", "zz_a", "zzx"]

    @needs_sqlalchemy
    def test_sqlalchemy(self, sqlalchemy_engine, escape_docs):
        t = sa.table(ESCAPE_COLLECTION, sa.column("_id"), sa.column("n"), sa.column("name"))
        like = t.c.name.like("zz%", escape="/")
        with sqlalchemy_engine.connect() as c:
            query = sa.select(t.c._id).where(sa.or_(sa.and_(t.c.n == 1, like), t.c.n == 2)).order_by(t.c._id)
            assert list(c.execute(query).scalars()) == [1, 2, 5]
            query = sa.select(t.c._id).where(sa.and_(sa.not_(like), t.c.n == 2)).order_by(t.c._id)
            assert list(c.execute(query).scalars()) == [2]
