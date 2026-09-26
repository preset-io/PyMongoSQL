# -*- coding: utf-8 -*-
"""GROUP BY, IN/NOT IN, LIKE, quoted literals and keyword aliases must return correct rows."""

import json

import pytest

from pymongosql.sql.parser import SQLParser
from tests.conftest import HAS_SQLALCHEMY, make_superset_conn

COLLECTION = "test_grouping_filters"
DOCS = [
    {"_id": 1, "flag": True, "dept": "a", "amount": 10, "name": "O'Brien"},
    {"_id": 2, "flag": False, "dept": "a", "amount": 20, "name": "x.y"},
    {"_id": 3, "flag": True, "dept": "b", "amount": 40, "name": "xzy"},
    {"_id": 4, "flag": True, "dept": "b", "amount": None, "name": "plain"},
]


def plan(sql):
    return SQLParser(sql).get_execution_plan()


def pipeline(sql):
    return json.loads(plan(sql).aggregate_pipeline)


class TestPlans:
    def test_group_by_groups_on_the_key(self):
        stages = pipeline("SELECT flag, COUNT(*) AS n FROM t GROUP BY flag")
        assert stages[0]["$group"]["_id"] == {"g0": "$flag"}
        assert stages[1]["$project"] == {"_id": 0, "flag": "$_id.g0", "n": 1}

    def test_aggregate_query_keeps_order_by_skip_and_limit(self):
        stages = pipeline("SELECT dept, SUM(amount) AS total FROM t GROUP BY dept ORDER BY total DESC LIMIT 1 OFFSET 1")
        assert stages[-3:] == [{"$sort": {"total": -1}}, {"$skip": 1}, {"$limit": 1}]

    def test_order_by_aggregate_expression_uses_its_output(self):
        stages = pipeline("SELECT dept, SUM(amount) AS total FROM t GROUP BY dept ORDER BY SUM(amount)")
        assert stages[-1] == {"$sort": {"total": 1}}

    def test_quoted_keyword_alias_is_unquoted(self):
        p = plan('SELECT flag, COUNT(*) AS "count" FROM t GROUP BY flag ORDER BY "count" DESC')
        assert list(p.projection_stage) == ["flag", "count"]
        assert json.loads(p.aggregate_pipeline)[-1] == {"$sort": {"count": -1}}

    def test_find_order_by_alias_sorts_on_the_field(self):
        p = plan('SELECT amount AS "value" FROM t ORDER BY "value" DESC')
        assert p.sort_stage == [{"amount": -1}]
        assert p.column_aliases == {"amount": "value"}

    def test_ungrouped_column_is_rejected(self):
        with pytest.raises(Exception, match="must appear in GROUP BY"):
            plan("SELECT flag, COUNT(*) FROM t")

    def test_having_filters_groups(self):
        stages = pipeline("SELECT flag, COUNT(*) AS n FROM t GROUP BY flag HAVING COUNT(*) > 1")
        assert stages[2] == {"$match": {"n": {"$gt": 1}}}

    def test_in_keeps_literal_types_and_quoted_commas(self):
        p = plan("SELECT _id FROM t WHERE _id IN (1, 2.5, 'a,b', 'it''s', TRUE, ?)")
        assert p.filter_stage == {"_id": {"$in": [1, 2.5, "a,b", "it's", True, {"$pymongosqlParam": True}]}}

    def test_not_in(self):
        assert plan("SELECT _id FROM t WHERE _id NOT IN (1, 2)").filter_stage == {"_id": {"$nin": [1, 2, None]}}

    def test_field_ending_in_not_is_not_negated(self):
        assert plan("SELECT _id FROM t WHERE cannot IN (1)").filter_stage == {"cannot": {"$in": [1]}}

    def test_not_like_and_regex_metacharacters(self):
        p = plan("SELECT _id FROM t WHERE name NOT LIKE 'x.%'")
        assert p.filter_stage == {"$and": [{"name": {"$not": {"$regex": "^x\\..*"}}}, {"name": {"$ne": None}}]}

    def test_double_dash_inside_literal_is_not_a_comment(self):
        p = plan("SELECT _id FROM t WHERE name = 'a -- b' AND n = 1 -- trailing comment")
        assert p.filter_stage == {"$and": [{"name": "a -- b"}, {"n": 1}]}

    def test_escaped_quote_in_string_literal(self):
        assert plan("SELECT _id FROM t WHERE name = 'O''Brien'").filter_stage == {"name": "O'Brien"}


@pytest.mark.skipif(not HAS_SQLALCHEMY, reason="SQLAlchemy not available")
class TestDialectQuoting:
    def test_keyword_alias_and_column_are_quoted(self):
        import sqlalchemy as sa

        from pymongosql.sqlalchemy_mongodb.sqlalchemy_dialect import PyMongoSQLDialect

        t = sa.table("t", sa.column("flag"), sa.column("value"))
        count = sa.func.count().label("count")
        stmt = sa.select(sa.column("flag"), t.c.value, count).select_from(t).group_by(sa.column("flag")).order_by(count)
        sql = " ".join(str(stmt.compile(dialect=PyMongoSQLDialect())).split())
        assert sql == 'SELECT flag, "value", count(*) AS "count" FROM t GROUP BY flag ORDER BY "count"'


@pytest.fixture
def grouping_collection(conn):
    conn.database.drop_collection(COLLECTION)
    conn.database[COLLECTION].insert_many(DOCS)
    yield conn
    conn.database.drop_collection(COLLECTION)


def rows(conn, sql, params=None):
    cursor = conn.cursor()
    cursor.execute(sql, params) if params is not None else cursor.execute(sql)
    return [tuple(r) for r in cursor.fetchall()]


class TestLive:
    def test_group_by_returns_one_row_per_group(self, grouping_collection):
        got = rows(grouping_collection, f"SELECT flag, COUNT(*) AS n FROM {COLLECTION} GROUP BY flag ORDER BY flag")
        assert got == [(False, 1), (True, 3)]

    def test_group_by_with_sum_where_parameter_order_and_limit(self, grouping_collection):
        sql = (
            f"SELECT dept, SUM(amount) AS total, COUNT(amount) AS counted FROM {COLLECTION} "
            "WHERE _id > ? GROUP BY dept ORDER BY total DESC LIMIT 1"
        )
        assert rows(grouping_collection, sql, [0]) == [("b", 40, 1)]

    def test_quoted_count_alias(self, grouping_collection):
        sql = f'SELECT flag, COUNT(*) AS "count" FROM {COLLECTION} GROUP BY flag ORDER BY "count" DESC'
        cursor = grouping_collection.cursor()
        cursor.execute(sql)
        assert [d[0] for d in cursor.description] == ["flag", "count"]
        assert [tuple(r) for r in cursor.fetchall()] == [(True, 3), (False, 1)]

    def test_in_and_not_in(self, grouping_collection):
        assert rows(grouping_collection, f"SELECT _id FROM {COLLECTION} WHERE _id IN (1, 3) ORDER BY _id") == [
            (1,),
            (3,),
        ]
        assert rows(grouping_collection, f"SELECT _id FROM {COLLECTION} WHERE _id IN (?, ?) ORDER BY _id", [2, 4]) == [
            (2,),
            (4,),
        ]
        assert rows(grouping_collection, f"SELECT _id FROM {COLLECTION} WHERE _id NOT IN (1, 3) ORDER BY _id") == [
            (2,),
            (4,),
        ]

    def test_escaped_quote_and_like(self, grouping_collection):
        assert rows(grouping_collection, f"SELECT _id FROM {COLLECTION} WHERE name = 'O''Brien'") == [(1,)]
        assert rows(grouping_collection, f"SELECT _id FROM {COLLECTION} WHERE name LIKE 'x.%'") == [(2,)]
        assert rows(grouping_collection, f"SELECT _id FROM {COLLECTION} WHERE name NOT LIKE 'x%' ORDER BY _id") == [
            (1,),
            (4,),
        ]

    def test_superset_mode_physical_table_chart_query(self, grouping_collection):
        conn = make_superset_conn()
        try:
            sql = (
                f'SELECT flag AS flag, COUNT(*) AS "count" FROM {COLLECTION} '
                'GROUP BY flag ORDER BY "count" DESC LIMIT 100'
            )
            assert rows(conn, sql) == [(True, 3), (False, 1)]
        finally:
            conn.close()

    def test_having_returns_only_matching_groups(self, grouping_collection):
        got = rows(grouping_collection, f"SELECT flag, COUNT(*) AS n FROM {COLLECTION} GROUP BY flag HAVING n > 1")
        assert got == [(True, 3)]


@pytest.mark.skipif(not HAS_SQLALCHEMY, reason="SQLAlchemy not available")
class TestLiveSQLAlchemy:
    def test_core_group_by_count_label(self, sqlalchemy_engine, grouping_collection):
        import sqlalchemy as sa

        t = sa.table(COLLECTION, sa.column("flag"), sa.column("_id"))
        count = sa.func.count().label("count")
        stmt = sa.select(t.c.flag, count).where(t.c._id.in_([1, 2, 3])).group_by(t.c.flag).order_by(count.desc())
        with sqlalchemy_engine.connect() as connection:
            assert [tuple(r) for r in connection.execute(stmt)] == [(True, 2), (False, 1)]
