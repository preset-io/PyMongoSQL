# -*- coding: utf-8 -*-
"""Superset-mode decimals, DATE_TRUNC time grains and literal binds.

- The superset-mode SQLite stage evaluates decimal columns exactly (SUM/AVG/MIN/MAX,
  GROUP BY, ORDER BY, comparisons) instead of as doubles or text.
- DATE_TRUNC(unit, field) is translated to MongoDB's $dateTrunc on a collection and
  evaluated with the same meaning in the SQLite stage.
- Compiling with literal_binds keeps a string literal containing ``%(name)s`` intact.
"""

import datetime
from decimal import Decimal, localcontext

import pytest
from bson import Decimal128

from pymongosql.error import Error, NotSupportedError
from pymongosql.sql.parser import SQLParser
from pymongosql.superset_mongodb.query_db_sqlite import QueryDBSQLite
from pymongosql.superset_mongodb.time_grain import UNITS, truncate
from tests.conftest import HAS_SQLALCHEMY, make_superset_conn

if HAS_SQLALCHEMY:
    import sqlalchemy as sa

    from pymongosql.sqlalchemy_mongodb.sqlalchemy_dialect import PyMongoSQLDialect

needs_sqlalchemy = pytest.mark.skipif(not HAS_SQLALCHEMY, reason="SQLAlchemy not available")
sqlglot = pytest.importorskip("sqlglot")

COLLECTION = "test_decimals_time_grains"
BIG = Decimal("123456789012345678901.1234567891")
AMOUNTS = [BIG, Decimal("0.0000000001"), Decimal("-5.5"), Decimal("10"), None, Decimal("9.99")]
GROUPS = ["a", "a", "b", "b", "b", "a"]
TIMES = [
    datetime.datetime(2025, 12, 31, 23, 59, 59, 999000),  # Wednesday
    datetime.datetime(2026, 1, 3, 10, 15, 30, 250000),  # Saturday
    datetime.datetime(2026, 1, 4, 0, 0, 0),  # Sunday
    datetime.datetime(2026, 1, 5, 8, 30, 0),  # Monday
    datetime.datetime(2026, 4, 1, 12, 0, 1),
    datetime.datetime(2026, 9, 26, 18, 45, 12, 5000),
]


def records():
    return [{"g": g, "amt": a, "ts": t} for g, a, t in zip(GROUPS, AMOUNTS, TIMES)]


def stage(sql):
    db = QueryDBSQLite()
    try:
        db.insert_records("virtual_table", records())
        return [tuple(r.values()) for r in db.execute_query(sql)]
    finally:
        db.close()


A_SUM = Decimal("123456789012345678911.1134567892")  # BIG + 0.0000000001 + 9.99, exact


def exact(values):
    return [v for v in values if v is not None]


class TestExactDecimalsInTheSQLiteStage:
    def test_sum_avg_min_max_are_exact(self):
        rows = stage('SELECT SUM(amt) AS "SUM(amt)", AVG(amt) AS a, MIN(amt) AS lo, MAX(amt) AS hi FROM virtual_table')
        # Decimal128 arithmetic: 34 significant digits, as MongoDB's own $sum and $avg
        with localcontext() as ctx:
            ctx.prec = 34
            total = sum(exact(AMOUNTS))
            average = total / 5
        assert total == Decimal("123456789012345678915.6134567892")
        assert rows == [(total, average, Decimal("-5.5"), BIG)]
        assert all(type(v) is Decimal for v in rows[0])

    def test_group_by_with_exact_sums(self):
        rows = stage('SELECT g AS g, SUM(amt) AS "SUM(amt)" FROM virtual_table GROUP BY g ORDER BY g')
        assert rows == [("a", A_SUM), ("b", Decimal("4.5"))]

    def test_group_by_the_decimal_column(self):
        rows = stage("SELECT amt AS amt, COUNT(*) AS n FROM virtual_table GROUP BY amt ORDER BY amt")
        assert [r[0] for r in rows] == [
            None,
            Decimal("-5.5"),
            Decimal("0.0000000001"),
            Decimal("9.99"),
            Decimal("10"),
            BIG,
        ]

    def test_order_by_is_numeric_not_text(self):
        rows = stage("SELECT amt FROM virtual_table WHERE amt IS NOT NULL ORDER BY amt DESC")
        assert [r[0] for r in rows] == [BIG, Decimal("10"), Decimal("9.99"), Decimal("0.0000000001"), Decimal("-5.5")]

    def test_top_n_by_aggregate(self):
        rows = stage('SELECT g, SUM(amt) AS "SUM(amt)" FROM virtual_table GROUP BY g ORDER BY "SUM(amt)" ASC LIMIT 1')
        assert rows == [("b", Decimal("4.5"))]

    def test_comparisons_with_numeric_literals_are_exact(self):
        # As doubles, 123456789012345678901.1234567891 equals 123456789012345678901.1234567890
        assert stage("SELECT COUNT(*) FROM virtual_table WHERE amt > 123456789012345678901.123456789") == [(1,)]
        assert stage("SELECT COUNT(*) FROM virtual_table WHERE amt = 123456789012345678901.123456789") == [(0,)]
        assert stage("SELECT COUNT(*) FROM virtual_table WHERE 0.00000000005 < amt AND amt < 1") == [(1,)]
        rows = stage('SELECT g, SUM(amt) AS "SUM(amt)" FROM virtual_table GROUP BY g HAVING SUM(amt) > 4.5')
        assert [r[0] for r in rows] == ["a"]

    def test_select_star_returns_decimals(self):
        rows = stage("SELECT * FROM virtual_table WHERE g = 'b' ORDER BY amt")
        assert [r[1] for r in rows] == [None, Decimal("-5.5"), Decimal("10")]

    def test_count_and_null_checks_are_allowed(self):
        assert stage("SELECT COUNT(amt), COUNT(*) FROM virtual_table WHERE amt IS NULL OR amt IS NOT NULL") == [(5, 6)]

    @pytest.mark.parametrize(
        "sql",
        [
            "SELECT amt * 2 FROM virtual_table",
            "SELECT ROUND(amt, 2) FROM virtual_table",
            "SELECT SUM(DISTINCT amt) FROM virtual_table",
            "SELECT * FROM (SELECT amt FROM virtual_table) AS x",
        ],
    )
    def test_other_uses_are_refused_rather_than_approximated(self, sql):
        with pytest.raises(NotSupportedError):
            stage(sql)

    def test_integer_and_float_columns_are_unchanged(self):
        db = QueryDBSQLite()
        try:
            db.insert_records("virtual_table", [{"i": 1, "f": 0.5}, {"i": 2, "f": 1.25}])
            assert db.execute_query("SELECT SUM(i) AS i, SUM(f) AS f FROM virtual_table") == [{"i": 3, "f": 1.75}]
        finally:
            db.close()


class TestDateTruncTranslation:
    def plan(self, sql):
        return SQLParser(sql).get_execution_plan()

    def test_grouped_time_grain(self):
        plan = self.plan(
            "SELECT DATE_TRUNC('month', ts) AS __timestamp, COUNT(*) AS \"count\" FROM t "
            "GROUP BY DATE_TRUNC('month', ts) ORDER BY \"count\" DESC LIMIT 10"
        )
        assert '"$dateTrunc": {"date": "$ts", "unit": "month"}' in plan.aggregate_pipeline

    def test_week_ending_adds_six_days(self):
        plan = self.plan(
            "SELECT DATE_TRUNC('week_ending_sunday', t.ts) AS x FROM t GROUP BY DATE_TRUNC('week_ending_sunday', t.ts)"
        )
        assert '"startOfWeek": "monday"' in plan.aggregate_pipeline
        assert '"$dateAdd"' in plan.aggregate_pipeline and '"$t.ts"' not in plan.aggregate_pipeline

    def test_ungrouped_projection(self):
        plan = self.plan("SELECT DATE_TRUNC('year', ts) AS y, name FROM t ORDER BY y DESC")
        assert '"$addFields"' in plan.aggregate_pipeline and '"$sort": {"__computed0": -1}' in plan.aggregate_pipeline

    @pytest.mark.parametrize(
        "sql", ["SELECT DATE_TRUNC('fortnight', ts) FROM t", "SELECT DATE_TRUNC(ts, 'day') FROM t"]
    )
    def test_invalid_date_trunc_is_refused(self, sql):
        with pytest.raises(Error):
            self.plan(sql)

    @pytest.mark.parametrize("unit", UNITS)
    def test_sqlite_stage_date_trunc(self, unit):
        rows = stage(f"SELECT DATE_TRUNC('{unit}', ts) AS t FROM virtual_table ORDER BY ts")
        assert [r[0] for r in rows] == [truncate(unit, t) for t in TIMES]


@needs_sqlalchemy
class TestLiteralBinds:
    def compile(self, stmt, **kw):
        return " ".join(str(stmt.compile(dialect=PyMongoSQLDialect(), **kw)).split())

    def test_percent_name_literal_is_kept(self):
        t = sa.table("t", sa.column("x"), sa.column("n"))
        stmt = sa.select(t.c.x).where(t.c.x == "%(k)s", t.c.n == 5)
        literal = self.compile(stmt, compile_kwargs={"literal_binds": True})
        assert literal == "SELECT x FROM t WHERE x = '%(k)s' AND n = 5"
        bound = stmt.compile(dialect=PyMongoSQLDialect())
        assert " ".join(str(bound).split()) == "SELECT x FROM t WHERE x = ? AND n = ?"
        assert list(bound.positiontup) == ["x_1", "n_1"]

    def test_quotes_inside_literals(self):
        t = sa.table("t", sa.column("x"))
        stmt = sa.select(t.c.x).where(t.c.x == 'it\'s %(a)s "q"')
        assert self.compile(stmt, compile_kwargs={"literal_binds": True}) == (
            "SELECT x FROM t WHERE x = 'it''s %(a)s \"q\"'"
        )

    def test_limit_with_literal_binds(self):
        t = sa.table("t", sa.column("x"))
        stmt = sa.select(t.c.x).where(t.c.x.contains("%(k)s", autoescape=True)).limit(3).offset(1)
        assert self.compile(stmt, compile_kwargs={"literal_binds": True}) == (
            "SELECT x FROM t WHERE (x LIKE '%' || '/%(k)s' || '%' ESCAPE '/') LIMIT 3 OFFSET 1"
        )


@pytest.fixture
def grain_docs(conn):
    conn.database.drop_collection(COLLECTION)
    conn.database[COLLECTION].insert_many(
        [
            {"_id": i, "g": g, "amt": None if a is None else Decimal128(a), "ts": t}
            for i, (g, a, t) in enumerate(zip(GROUPS, AMOUNTS, TIMES))
        ]
    )
    yield conn
    conn.database.drop_collection(COLLECTION)


class TestLive:
    @pytest.mark.parametrize("unit", UNITS)
    def test_physical_time_grain(self, grain_docs, unit):
        cursor = grain_docs.cursor()
        cursor.execute(
            f"SELECT DATE_TRUNC('{unit}', ts) AS __timestamp, COUNT(*) AS \"count\" FROM {COLLECTION} "
            f"WHERE ts >= STR_TO_DATETIME('2025-01-01T00:00:00') GROUP BY DATE_TRUNC('{unit}', ts) "
            "ORDER BY __timestamp"
        )
        expected = {}
        for t in TIMES:
            expected[truncate(unit, t)] = expected.get(truncate(unit, t), 0) + 1
        assert [tuple(r) for r in cursor.fetchall()] == sorted(expected.items())

    def test_physical_ungrouped_time_grain(self, grain_docs):
        cursor = grain_docs.cursor()
        cursor.execute(f"SELECT DATE_TRUNC('week', ts) AS w, g FROM {COLLECTION} ORDER BY w DESC LIMIT 2")
        assert [tuple(r) for r in cursor.fetchall()] == [
            (datetime.datetime(2026, 9, 20), "a"),
            (datetime.datetime(2026, 3, 29), "b"),
        ]

    @pytest.mark.parametrize("unit", UNITS)
    def test_virtual_time_grain(self, grain_docs, unit):
        superset = make_superset_conn()
        try:
            cursor = superset.cursor()
            cursor.execute(
                f"SELECT DATE_TRUNC('{unit}', ts) AS __timestamp, COUNT(*) AS \"count\" "
                f"FROM (SELECT ts FROM {COLLECTION}) AS virtual_table "
                "WHERE ts >= STR_TO_DATETIME('2025-01-01T00:00:00') "
                f"GROUP BY DATE_TRUNC('{unit}', ts) ORDER BY __timestamp"
            )
            expected = {}
            for t in TIMES:
                expected[truncate(unit, t)] = expected.get(truncate(unit, t), 0) + 1
            assert [tuple(r) for r in cursor.fetchall()] == sorted(expected.items())
        finally:
            superset.close()

    def test_virtual_dataset_decimals_are_exact(self, grain_docs):
        superset = make_superset_conn()
        try:
            cursor = superset.cursor()
            cursor.execute(
                'SELECT g AS g, SUM(amt) AS "SUM(amt)" '
                f'FROM (SELECT g, amt FROM {COLLECTION}) AS virtual_table GROUP BY g ORDER BY "SUM(amt)" DESC'
            )
            assert [tuple(r) for r in cursor.fetchall()] == [
                ("a", A_SUM),
                ("b", Decimal("4.5")),
            ]
            cursor.execute(f"SELECT amt FROM (SELECT amt FROM {COLLECTION}) AS virtual_table ORDER BY amt DESC LIMIT 3")
            assert [r[0] for r in cursor.fetchall()] == [BIG, Decimal("10"), Decimal("9.99")]
        finally:
            superset.close()

    @needs_sqlalchemy
    def test_literal_binds_statement_runs(self, sqlalchemy_engine, grain_docs):
        grain_docs.database[COLLECTION].insert_one({"_id": 99, "g": "%(k)s"})
        t = sa.table(COLLECTION, sa.column("_id"), sa.column("g"))
        stmt = sa.select(t.c._id).where(t.c.g == "%(k)s")
        sql = str(stmt.compile(sqlalchemy_engine, compile_kwargs={"literal_binds": True}))
        with sqlalchemy_engine.connect() as c:
            assert list(c.exec_driver_sql(sql).scalars()) == [99]
            assert list(c.execute(stmt).scalars()) == [99]


class TestLiveAggregateNullSemantics:
    @pytest.fixture
    def null_docs(self, conn):
        name = COLLECTION + "_nulls"
        conn.database.drop_collection(name)
        conn.database[name].insert_many(
            [
                {"_id": 1, "g": "a", "amt": None},
                {"_id": 2, "g": "b", "amt": Decimal128("1.5")},
                {"_id": 3, "g": "b"},
                {"_id": 4, "g": "c", "amt": 0},
            ]
        )
        yield conn, name
        conn.database.drop_collection(name)

    def rows(self, conn, sql):
        cursor = conn.cursor()
        cursor.execute(sql)
        return [tuple(r) for r in cursor.fetchall()]

    def test_sum_of_no_values_is_null(self, null_docs):
        conn, name = null_docs
        rows = self.rows(conn, f"SELECT g, SUM(amt) AS s, SUM(DISTINCT amt) AS d FROM {name} GROUP BY g ORDER BY g")
        assert rows == [("a", None, None), ("b", Decimal("1.5"), Decimal("1.5")), ("c", 0, 0)]
        assert self.rows(conn, f"SELECT SUM(amt) AS s FROM {name} WHERE g = 'a'") == [(None,)]

    def test_aggregate_without_group_by_returns_one_row_for_no_input(self, null_docs):
        conn, name = null_docs
        empty = f"FROM {name} WHERE g = 'none'"
        assert self.rows(
            conn, f"SELECT SUM(amt) AS s, COUNT(*) AS n, COUNT(DISTINCT g) AS d, MAX(amt) AS m {empty}"
        ) == [(None, 0, 0, None)]
        assert self.rows(conn, f"SELECT COUNT(*) AS n {empty} HAVING COUNT(*) > 0") == []
        assert self.rows(conn, f"SELECT g, COUNT(*) AS n {empty} GROUP BY g") == []
        assert self.rows(conn, f"SELECT COUNT(*) AS n FROM {name} LIMIT 0") == []


class TestLiveEmptyVirtualDataset:
    def test_inner_query_without_rows(self, grain_docs):
        superset = make_superset_conn()
        inner = f"(SELECT g, amt FROM {COLLECTION} WHERE g = 'none') AS virtual_table"
        try:
            cursor = superset.cursor()
            cursor.execute(f'SELECT COUNT(*) AS "count", SUM(amt) AS s FROM {inner}')
            assert [tuple(r) for r in cursor.fetchall()] == [(0, None)]
            cursor.execute(f'SELECT g, COUNT(*) AS "count" FROM {inner} GROUP BY g')
            assert cursor.fetchall() == []
            cursor.execute(f"SELECT g FROM {inner}")
            assert cursor.fetchall() == [] and [d[0] for d in cursor.description] == ["g"]
        finally:
            superset.close()
