# -*- coding: utf-8 -*-
"""Superset-mode subqueries keep booleans and nullable numbers typed through SQLite."""

from pymongosql.superset_mongodb.query_db_sqlite import QueryDBSQLite, SQLiteTypeMapper
from tests.conftest import make_superset_conn

COLLECTION = "test_superset_types"
DOCS = [
    {"_id": 1, "flag": True, "i": 5, "n": None},
    {"_id": 2, "flag": False, "i": -7, "n": 2},
    {"_id": 3, "flag": None, "i": None, "n": 3},
]


def query(sql, records=None):
    db = QueryDBSQLite()
    try:
        db.insert_records("v", records or [{k: v for k, v in d.items() if k != "_id"} for d in DOCS])
        return db.execute_query(sql)
    finally:
        db.close()


class TestSQLiteStage:
    def test_selected_boolean_column_is_bool(self):
        got = query("SELECT flag AS flag, SUM(i) AS s FROM v GROUP BY flag ORDER BY s DESC")
        assert got == [{"flag": True, "s": 5}, {"flag": False, "s": -7}, {"flag": None, "s": None}]
        assert [type(r["flag"]) for r in got] == [bool, bool, type(None)]

    def test_expression_over_boolean_stays_numeric(self):
        assert query("SELECT SUM(flag) AS n FROM v") == [{"n": 1}]

    def test_null_does_not_turn_a_column_into_text(self):
        assert SQLiteTypeMapper.infer_schema([{"n": None}, {"n": 2}, {"n": None}]) == {"n": "INTEGER"}
        got = query("SELECT n FROM v ORDER BY n")
        assert got == [{"n": None}, {"n": 2}, {"n": 3}]
        assert type(got[1]["n"]) is int

    def test_mixed_types_still_fall_back_to_text(self):
        assert SQLiteTypeMapper.infer_schema([{"x": 1}, {"x": "a"}]) == {"x": "TEXT"}


class TestLive:
    def test_virtual_dataset_returns_booleans(self, conn):
        conn.database.drop_collection(COLLECTION)
        conn.database[COLLECTION].insert_many(DOCS)
        superset = make_superset_conn()
        try:
            cursor = superset.cursor()
            cursor.execute(
                'SELECT flag AS flag, SUM(i) AS "sum_i" FROM '
                f"(SELECT flag, i FROM {COLLECTION}) AS virtual_table "
                "GROUP BY flag ORDER BY flag DESC"
            )
            rows = [tuple(r) for r in cursor.fetchall()]
            assert rows == [(True, 5), (False, -7), (None, None)]
            assert type(rows[0][0]) is bool and type(rows[1][0]) is bool
            cursor.execute(f"SELECT n FROM (SELECT n FROM {COLLECTION}) AS virtual_table ORDER BY n")
            values = [r[0] for r in cursor.fetchall()]
            assert values == [None, 2, 3] and type(values[1]) is int
        finally:
            superset.close()
            conn.database.drop_collection(COLLECTION)
