# -*- coding: utf-8 -*-
"""Quoted operator names, LIKE ... ESCAPE precedence and paging of aggregation pipelines.

- A field name starting with ``$`` is rejected whether or not it is quoted, so
  ``"$where"`` or ``"$expr"`` can never reach MongoDB as a query operator.
- ``LIKE ... ESCAPE`` keeps SQL precedence for the conditions that follow it: the
  grammar parses the rest of the boolean chain as the escape expression.
- LIMIT and OFFSET of a GROUP BY, aggregate or DATE_TRUNC query become ``$skip`` and
  ``$limit`` stages, so the server returns only the requested page.
"""

import datetime
import json

import pytest
from pymongo import monitoring

from pymongosql.error import Error
from pymongosql.sql.parser import SQLParser
from tests.conftest import HAS_SQLALCHEMY, make_conn, make_superset_conn

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

    @pytest.mark.parametrize("key", ['"$ROOT"', '"$where"', '"$expr"', '"$"', 'a."$b"', '"a.$b"', "a['$c']"])
    def test_group_by_key_is_rejected(self, key):
        with pytest.raises(Error):
            plan(f"SELECT COUNT(*) AS c FROM t GROUP BY {key}")

    def test_group_by_names_without_leading_dollar_still_work(self):
        for key, field in (('"price$"', "price$"), ('"a$b"', "a$b"), ('"first name"', "first name")):
            pipeline = json.loads(plan(f"SELECT COUNT(*) AS c FROM t GROUP BY {key}").aggregate_pipeline)
            assert pipeline[0]["$group"]["_id"] == {"g0": f"${field}"}


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


# ------------------------------------------------------------ paging aggregation pipelines


class _AggregateListener(monitoring.CommandListener):
    """Records aggregate pipelines and how many documents the server returned."""

    def __init__(self):
        self.pipelines = []
        self.returned = 0

    def started(self, event):
        if event.command_name == "aggregate":
            self.pipelines.append(event.command["pipeline"])

    def succeeded(self, event):
        if event.command_name in ("aggregate", "getMore"):
            batch = event.reply.get("cursor", {})
            self.returned += len(batch.get("firstBatch", batch.get("nextBatch", [])))

    def failed(self, event):
        pass


PAGING_COLLECTION = "test_aggregate_paging"
PAGING_DOCS = 500
START = datetime.datetime(2026, 1, 1)


@pytest.fixture
def paging(conn):
    conn.database.drop_collection(PAGING_COLLECTION)
    conn.database[PAGING_COLLECTION].insert_many(
        [{"_id": i, "s": f"k{i:03d}", "v": i, "ts": START + datetime.timedelta(days=i)} for i in range(PAGING_DOCS)]
    )
    listener = _AggregateListener()
    monitored = make_conn(event_listeners=[listener])
    try:
        yield monitored, listener
    finally:
        monitored.close()
        conn.database.drop_collection(PAGING_COLLECTION)


@pytest.fixture
def superset_paging(paging):
    _, listener = paging
    monitored = make_superset_conn(event_listeners=[listener])
    try:
        yield monitored, listener
    finally:
        monitored.close()


def rows(conn, sql, params=None):
    return [tuple(r) for r in run(conn, sql, params).fetchall()]


class TestLiveAggregatePaging:
    def test_group_by_limit_offset(self, paging):
        conn, listener = paging
        sql = f"SELECT s, COUNT(*) AS c FROM {PAGING_COLLECTION} GROUP BY s ORDER BY s LIMIT 3 OFFSET 2"
        assert rows(conn, sql) == [("k002", 1), ("k003", 1), ("k004", 1)]
        assert listener.pipelines[-1][-2:] == [{"$skip": 2}, {"$limit": 3}]
        assert listener.returned == 3

    def test_bound_limit_offset(self, paging):
        conn, listener = paging
        sql = f"SELECT s, SUM(v) AS t FROM {PAGING_COLLECTION} GROUP BY s ORDER BY s LIMIT ? OFFSET ?"
        assert rows(conn, sql, [2, 1]) == [("k001", 1), ("k002", 2)]
        assert listener.pipelines[-1][-2:] == [{"$skip": 1}, {"$limit": 2}]
        assert listener.returned == 2

    def test_date_trunc_limit(self, paging):
        conn, listener = paging
        sql = f"SELECT DATE_TRUNC('day', ts) AS d, v FROM {PAGING_COLLECTION} ORDER BY v LIMIT 3"
        assert rows(conn, sql) == [(START + datetime.timedelta(days=i), i) for i in range(3)]
        assert listener.pipelines[-1][-1] == {"$limit": 3}
        assert listener.returned == 3

    def test_aggregate_without_group_by(self, paging):
        conn, listener = paging
        assert rows(conn, f"SELECT COUNT(*) AS c FROM {PAGING_COLLECTION} LIMIT 1 OFFSET 1") == []
        assert listener.returned == 0

    def test_limit_zero(self, paging):
        conn, listener = paging
        assert rows(conn, f"SELECT s, COUNT(*) AS c FROM {PAGING_COLLECTION} GROUP BY s LIMIT 0") == []
        assert listener.returned <= 1

    def test_offset_only(self, paging):
        conn, listener = paging
        sql = f"SELECT s, COUNT(*) AS c FROM {PAGING_COLLECTION} GROUP BY s ORDER BY s OFFSET 498"
        assert rows(conn, sql) == [("k498", 1), ("k499", 1)]
        assert listener.returned == 2

    @pytest.mark.parametrize("huge", [2**63, 2**63 + 5, 2**70])
    def test_limit_beyond_int64_pages_in_python(self, paging, huge):
        conn, listener = paging
        sql = f"SELECT s, COUNT(*) AS c FROM {PAGING_COLLECTION} GROUP BY s ORDER BY s LIMIT {huge}"
        assert len(rows(conn, sql)) == PAGING_DOCS
        assert not any("$limit" in stage or "$skip" in stage for stage in listener.pipelines[-1])

    @pytest.mark.parametrize("huge", [2**63, 2**70])
    def test_offset_beyond_int64_pages_in_python(self, paging, huge):
        conn, listener = paging
        sql = f"SELECT s, COUNT(*) AS c FROM {PAGING_COLLECTION} GROUP BY s ORDER BY s LIMIT 3 OFFSET {huge}"
        assert rows(conn, sql) == []
        assert not any("$limit" in stage or "$skip" in stage for stage in listener.pipelines[-1])

    def test_bound_values_beyond_int64_page_in_python(self, paging):
        conn, listener = paging
        sql = f"SELECT s, COUNT(*) AS c FROM {PAGING_COLLECTION} GROUP BY s ORDER BY s LIMIT ? OFFSET ?"
        assert len(rows(conn, sql, [2**63, 497])) == 3
        assert rows(conn, sql, [3, 2**63]) == []
        assert not any("$limit" in stage or "$skip" in stage for stage in listener.pipelines[-1])

    def test_int64_maximum_is_still_pushed_down(self, paging):
        conn, listener = paging
        top = 2**63 - 1
        sql = f"SELECT s, COUNT(*) AS c FROM {PAGING_COLLECTION} GROUP BY s ORDER BY s LIMIT {top} OFFSET 497"
        assert rows(conn, sql) == [("k497", 1), ("k498", 1), ("k499", 1)]
        assert listener.pipelines[-1][-2:] == [{"$skip": 497}, {"$limit": top}]
        assert listener.returned == 3
        assert (
            rows(conn, f"SELECT s, COUNT(*) AS c FROM {PAGING_COLLECTION} GROUP BY s ORDER BY s LIMIT 1 OFFSET {top}")
            == []
        )
        assert listener.pipelines[-1][-2:] == [{"$skip": top}, {"$limit": 1}]

    def test_no_limit_returns_every_group(self, paging):
        conn, listener = paging
        assert len(rows(conn, f"SELECT s, COUNT(*) AS c FROM {PAGING_COLLECTION} GROUP BY s")) == PAGING_DOCS
        assert listener.returned == PAGING_DOCS

    def test_aggregate_function_without_where_or_order_by(self, paging):
        conn, listener = paging
        sql = f"SELECT * FROM {PAGING_COLLECTION}.aggregate('[{{\"$sort\": {{\"v\": 1}}}}]', '{{}}') LIMIT 2 OFFSET 3"
        assert [r[0] for r in rows(conn, sql)] == [3, 4]
        assert listener.returned == 2

    def test_aggregate_function_with_where_still_filters_before_paging(self, paging):
        conn, _ = paging
        sql = (
            f"SELECT * FROM {PAGING_COLLECTION}.aggregate('[{{\"$sort\": {{\"v\": 1}}}}]', '{{}}') "
            "WHERE v >= 10 LIMIT 2"
        )
        assert [r[0] for r in rows(conn, sql)] == [10, 11]

    def test_superset_mode(self, superset_paging):
        conn, listener = superset_paging
        sql = f"SELECT s, COUNT(*) AS c FROM {PAGING_COLLECTION} GROUP BY s ORDER BY s LIMIT 3"
        assert rows(conn, sql) == [("k000", 1), ("k001", 1), ("k002", 1)]
        assert listener.returned == 3
