# -*- coding: utf-8 -*-
"""Predicates are read from the parse tree: field names, operators and values are never
recovered by searching concatenated token text.

The DML tests check the exact documents a DELETE or UPDATE touches: a truncated
field name there removes or rewrites the wrong documents.
"""

from decimal import Decimal

import pytest
from bson import Decimal128, Int64

from pymongosql.error import Error
from pymongosql.sql.parser import SQLParser
from tests.conftest import HAS_SQLALCHEMY

if HAS_SQLALCHEMY:
    import sqlalchemy as sa

    from pymongosql.sqlalchemy_mongodb.sqlalchemy_dialect import PyMongoSQLDialect

needs_sqlalchemy = pytest.mark.skipif(not HAS_SQLALCHEMY, reason="SQLAlchemy not available")
PARAM = {"$pymongosqlParam": True}


def plan(sql):
    return SQLParser(sql).get_execution_plan()


def where(sql_where):
    return plan(f"SELECT _id FROM t WHERE {sql_where}").filter_stage


def compiled(stmt):
    return " ".join(str(stmt.compile(dialect=PyMongoSQLDialect())).split())


class TestFieldNamesAreNotTruncated:
    @pytest.mark.parametrize(
        "sql_where,expected",
        [
            ("dislikes IS NULL", {"dislikes": {"$eq": None}}),
            ("unlike = 5", {"unlike": 5}),
            ("likes IN (1, 2)", {"likes": {"$in": [1, 2]}}),
            ("isnull = 1", {"isnull": 1}),
            ("between_x BETWEEN 1 AND 2", {"$and": [{"between_x": {"$gte": 1}}, {"between_x": {"$lte": 2}}]}),
            ("login_count > 3", {"login_count": {"$gt": 3}}),
            ("inx <> 'a'", {"inx": {"$nin": ["a", None]}}),
        ],
    )
    def test_select(self, sql_where, expected):
        assert where(sql_where) == expected

    def test_delete_and_update(self):
        assert plan("DELETE FROM t WHERE dislikes IS NULL").filter_conditions == {"dislikes": {"$eq": None}}
        update = plan("UPDATE t SET x = 1 WHERE unlike = 5")
        assert update.filter_conditions == {"unlike": 5}

    def test_string_literal_containing_keywords(self):
        assert where("name = 'x IN(1) LIKE y BETWEEN'") == {"name": "x IN(1) LIKE y BETWEEN"}

    def test_reversed_operands(self):
        assert where("5 < age") == {"age": {"$gt": 5}}
        assert where("'x' = name") == {"name": "x"}
        assert where("10 >= age") == {"age": {"$lte": 10}}

    @pytest.mark.parametrize(
        "sql_where,field",
        [('"my field" = 1', "my field"), ('"a-b" = 1', "a-b"), ('"año" = 1', "año"), ('"a"."b c" = 1', "a.b c")],
    )
    def test_quoted_field_names(self, sql_where, field):
        assert where(sql_where) == {field: 1}

    @pytest.mark.parametrize("sql_where", ["a = b", "lower(a) = 'x'", "a + 1 = 2", "a IN (SELECT b FROM u)"])
    def test_untranslatable_predicates_raise(self, sql_where):
        with pytest.raises(Error):
            where(sql_where)


class TestLike:
    def test_concatenated_pattern(self):
        assert where("n LIKE '%' || 'ab' || '%'") == {"n": {"$regex": ".*ab.*"}}
        assert where("n LIKE 'ab' || '%'") == {"n": {"$regex": "^ab.*"}}

    def test_escape(self):
        assert where("n LIKE '50/%%' ESCAPE '/'") == {"n": {"$regex": "^50%.*"}}
        assert where("n LIKE 'a/_b' ESCAPE '/'") == {"n": {"$regex": "^a_b$"}}

    def test_escape_followed_by_more_conditions(self):
        assert where("n LIKE 'a/_%' ESCAPE '/' AND b = 1 OR c = 2") == {
            "$or": [{"$and": [{"n": {"$regex": "^a_.*"}}, {"b": 1}]}, {"c": 2}]
        }

    def test_inner_wildcards_match_newlines(self):
        assert where("n LIKE 'a%b'") == {"n": {"$regex": "^a.*b$", "$options": "s"}}

    def test_bound_pattern_is_translated_when_bound(self):
        from pymongosql.helper import SQLHelper

        f = where("n LIKE '%' || ? || '%' ESCAPE '/'")
        bound, used = SQLHelper.bind_filter(f, ["5/%(k)s"])
        assert (bound, used) == ({"n": {"$regex": ".*5%\\(k\\)s.*"}}, 1)
        with pytest.raises(Error):
            SQLHelper.bind_filter(where("n LIKE ?"), [5])

    def test_case_insensitive_like_from_lower(self):
        assert where("lower(n) LIKE lower('A%b')") == {"n": {"$regex": "^A.*b$", "$options": "si"}}
        assert where("lower(n) LIKE 'a%'") == {"n": {"$regex": "^a.*", "$options": "i"}}

    @needs_sqlalchemy
    def test_sqlalchemy_helpers_translate_after_binding(self):
        from pymongosql.helper import SQLHelper

        t = sa.table("t", sa.column("n"))
        for expr, regex in (
            (t.c.n.contains("ab"), {"$regex": ".*ab.*"}),
            (t.c.n.startswith("ab"), {"$regex": "^ab.*"}),
            (t.c.n.endswith("ab"), {"$regex": ".*ab$"}),
            (t.c.n.contains("5%_ %(k)s", autoescape=True), {"$regex": ".*5%_\\ %\\(k\\)s.*"}),
            (t.c.n.like("a/_%", escape="/"), {"$regex": "^a_.*"}),
            (t.c.n.ilike("A%"), {"$regex": "^A.*", "$options": "i"}),
        ):
            compiled_stmt = sa.select(t.c.n).where(expr).compile(dialect=PyMongoSQLDialect())
            params = [compiled_stmt.params[name] for name in compiled_stmt.positiontup]
            bound, _ = SQLHelper.bind_filter(plan(str(compiled_stmt)).filter_stage, params)
            assert bound == {"n": regex}, str(compiled_stmt)


class TestParameters:
    def test_literal_question_mark_is_a_value(self):
        assert where("a = '?' AND b = ?") == {"$and": [{"a": "?"}, {"b": PARAM}]}

    def test_literal_question_mark_in_aggregate(self):
        p = plan("SELECT g, COUNT(*) AS n FROM t WHERE a = '?' AND b = ? GROUP BY g")
        assert '"a": "?"' in p.aggregate_pipeline and '"$pymongosqlParam"' in p.aggregate_pipeline

    def test_limit_and_offset_parameters_are_kept(self):
        p = plan("SELECT _id FROM t WHERE a = ? LIMIT ? OFFSET ?")
        assert (p.filter_stage, p.limit_stage, p.skip_stage) == ({"a": PARAM}, PARAM, PARAM)

    @pytest.mark.parametrize("clause", ["LIMIT -1", "LIMIT 'x'", "OFFSET 1.5", "LIMIT a"])
    def test_invalid_limit_or_offset_raises_instead_of_returning_everything(self, clause):
        with pytest.raises(Error):
            plan(f"SELECT _id FROM t {clause}")

    @needs_sqlalchemy
    def test_sqlalchemy_limit_and_offset_are_literals(self):
        t = sa.table("t", sa.column("a"))
        assert compiled(sa.select(t.c.a).limit(5).offset(10)) == "SELECT a FROM t LIMIT 5 OFFSET 10"
        assert compiled(sa.select(t.c.a).offset(3)) == "SELECT a FROM t OFFSET 3"


class TestDecimalLiterals:
    def test_exact_decimal_literal_is_a_double(self):
        assert where("t > 36.5") == {"t": {"$gt": 36.5}}

    def test_inexact_decimal_literal_compares_in_the_field_type(self):
        assert where("a = 0.1") == {
            "$or": [
                {"$and": [{"a": {"$type": "double"}}, {"a": 0.1}]},
                {"$and": [{"a": {"$not": {"$type": "double"}}}, {"a": Decimal128("0.1")}]},
            ]
        }


class TestAggregates:
    def test_count_distinct(self):
        p = plan('SELECT g, COUNT(DISTINCT a) AS "COUNT_DISTINCT(a)" FROM t GROUP BY g')
        assert '"$addToSet": "$a"' in p.aggregate_pipeline
        assert '"COUNT_DISTINCT(a)": {"$size": {"$setDifference": ["$__agg0", [null]]}}' in p.aggregate_pipeline

    @pytest.mark.parametrize("sql", ["SELECT a + 1 FROM t", "SELECT lower(a) FROM t", "SELECT EVERY(a) FROM t"])
    def test_untranslatable_select_expressions_raise(self, sql):
        with pytest.raises(Error):
            plan(sql)

    def test_having(self):
        p = plan("SELECT g, COUNT(*) AS n FROM t GROUP BY g HAVING n > 1 AND SUM(v) >= 3")
        assert '{"$match": {"$and": [{"n": {"$gt": 1}}, {"__having0": {"$gte": 3}}]}}' in p.aggregate_pipeline
        assert '{"$project": {"__having0": 0}}' in p.aggregate_pipeline


@needs_sqlalchemy
class TestFloatColumns:
    """Float returns float; Float(asdecimal=True) returns Decimal as SQLAlchemy documents."""

    def processor(self, type_):
        dialect = PyMongoSQLDialect()
        return type_.dialect_impl(dialect).result_processor(dialect, None)

    def test_float_returns_float(self):
        for type_ in (sa.Float(), sa.FLOAT(), sa.REAL()):
            value = self.processor(type_)(Decimal128("1.25"))
            assert value == 1.25 and type(value) is float

    def test_float_asdecimal_returns_decimal(self):
        value = self.processor(sa.Float(asdecimal=True))(Decimal128("1.25"))
        assert value == Decimal("1.25") and type(value) is Decimal


# ---------------------------------------------------------------- live MongoDB

COLLECTION = "test_parse_tree_predicates"
DOCS = [
    {"_id": 1, "dislikes": None, "dis": 1, "unlike": 5, "un": "x", "n": "ab-50%", "g": "a", "v": 1, "f": 0.1},
    {"_id": 2, "dislikes": 3, "unlike": 6, "un": "=5", "n": "zab", "g": "a", "v": 2, "f": 0.2},
    {"_id": 3, "dislikes": 4, "dis": 2, "unlike": 5, "n": "a_b", "g": "b", "v": 2, "f": Decimal128("0.1")},
    {"_id": 4, "my field": "x", "año": 1, "n": "?", "g": "b", "v": None, "f": None},
]


@pytest.fixture
def docs(conn):
    conn.database.drop_collection(COLLECTION)
    conn.database[COLLECTION].insert_many([dict(d) for d in DOCS])
    yield conn
    conn.database.drop_collection(COLLECTION)


def run(conn, sql, params=None):
    cursor = conn.cursor()
    cursor.execute(sql, params) if params is not None else cursor.execute(sql)
    return cursor


def ids(conn, sql_where, params=None):
    return [r[0] for r in run(conn, f"SELECT _id FROM {COLLECTION} WHERE {sql_where} ORDER BY _id", params).fetchall()]


def remaining(conn):
    return [d["_id"] for d in conn.database[COLLECTION].find({}, sort=[("_id", 1)])]


class TestLiveDML:
    def test_delete_is_null_touches_only_null_dislikes(self, docs):
        # dislikes IS NULL: _id 1 (null) and 4 (missing); the old translation matched
        # every document lacking a field named "dis" (1 was spared, 2 and 4 deleted)
        run(docs, f"DELETE FROM {COLLECTION} WHERE dislikes IS NULL")
        assert remaining(docs) == [2, 3]

    def test_update_touches_only_matching_unlike(self, docs):
        run(docs, f"UPDATE {COLLECTION} SET v = 99 WHERE unlike = 5")
        assert {d["_id"]: d.get("v") for d in docs.database[COLLECTION].find()} == {1: 99, 2: 2, 3: 99, 4: None}

    def test_delete_with_parameters_and_literal_question_mark(self, docs):
        run(docs, f"DELETE FROM {COLLECTION} WHERE n = '?' OR unlike = ?", [6])
        assert remaining(docs) == [1, 3]

    def test_delete_reversed_operand(self, docs):
        run(docs, f"DELETE FROM {COLLECTION} WHERE 6 <= unlike")
        assert remaining(docs) == [1, 3, 4]

    def test_unused_parameter_refuses_the_delete(self, docs):
        with pytest.raises(Error):
            run(docs, f"DELETE FROM {COLLECTION} WHERE unlike = ?", [5, 6])
        assert remaining(docs) == [1, 2, 3, 4]

    def test_untranslatable_update_changes_nothing(self, docs):
        with pytest.raises(Error):
            run(docs, f"UPDATE {COLLECTION} SET v = 0 WHERE lower(n) = 'ab'")
        assert [d.get("v") for d in docs.database[COLLECTION].find({}, sort=[("_id", 1)])] == [1, 2, 2, None]


class TestLiveSelect:
    @pytest.mark.parametrize(
        "sql_where,expected",
        [
            ("dislikes IS NULL", [1, 4]),
            ("unlike = 5", [1, 3]),
            ("5 < unlike", [2]),
            ("\"my field\" = 'x'", [4]),
            ('"año" = 1', [4]),
            ("n LIKE '%' || 'ab' || '%'", [1, 2]),
            ("n LIKE '%/%' ESCAPE '/'", [1]),
            ("n LIKE 'a/_b' ESCAPE '/' AND g = 'b'", [3]),
            ("n = '?'", [4]),
            ("f = 0.1", [1, 3]),
        ],
    )
    def test_rows(self, docs, sql_where, expected):
        assert ids(docs, sql_where) == expected

    def test_bound_limit_offset_and_limit_zero(self, docs):
        page = run(docs, f"SELECT _id FROM {COLLECTION} WHERE v >= ? ORDER BY _id LIMIT ? OFFSET ?", [1, 2, 1])
        assert [r[0] for r in page.fetchall()] == [2, 3]
        assert run(docs, f"SELECT _id FROM {COLLECTION} LIMIT 0").fetchall() == []
        assert run(docs, f"SELECT g, COUNT(*) AS n FROM {COLLECTION} GROUP BY g LIMIT ?", [0]).fetchall() == []
        grouped = run(docs, f"SELECT g, COUNT(*) AS n FROM {COLLECTION} GROUP BY g ORDER BY g LIMIT ? OFFSET ?", [1, 1])
        assert [tuple(r) for r in grouped.fetchall()] == [("b", 2)]
        with pytest.raises(Error):
            run(docs, f"SELECT _id FROM {COLLECTION} LIMIT ?", [-1])
        with pytest.raises(Error):
            run(docs, f"SELECT _id FROM {COLLECTION} WHERE v = ?", [1, 2])

    def test_count_distinct_and_having(self, docs):
        rows = run(
            docs,
            f"SELECT g, COUNT(DISTINCT v) AS d, COUNT(*) AS n FROM {COLLECTION} GROUP BY g HAVING SUM(v) >= 2 ORDER BY g",
        ).fetchall()
        assert [tuple(r) for r in rows] == [("a", 2, 2), ("b", 1, 2)]
        rows = run(docs, f"SELECT g FROM {COLLECTION} GROUP BY g HAVING COUNT(*) > 1 AND MAX(v) > ?", [1]).fetchall()
        assert sorted(tuple(r) for r in rows) == [("a",), ("b",)]

    def test_decimal128_and_int64_are_python_types(self, docs):
        docs.database[COLLECTION].insert_one({"_id": 5, "d": Decimal128("1.10"), "i": Int64(2**40)})
        row = run(docs, f"SELECT d, i FROM {COLLECTION} WHERE _id = 5").fetchone()
        assert row == (Decimal("1.10"), 2**40)
        assert type(row[0]) is Decimal and type(row[1]) is int


@needs_sqlalchemy
class TestLiveSQLAlchemy:
    def test_like_helpers_limit_offset_and_count_distinct(self, sqlalchemy_engine, docs):
        t = sa.table(COLLECTION, sa.column("_id"), sa.column("n"), sa.column("g"), sa.column("v"))
        with sqlalchemy_engine.connect() as c:
            assert list(c.execute(sa.select(t.c._id).where(t.c.n.contains("ab")).order_by(t.c._id)).scalars()) == [1, 2]
            assert list(c.execute(sa.select(t.c._id).where(t.c.n.startswith("a")).order_by(t.c._id)).scalars()) == [
                1,
                3,
            ]
            assert list(c.execute(sa.select(t.c._id).where(t.c.n.endswith("b")).order_by(t.c._id)).scalars()) == [2, 3]
            assert list(c.execute(sa.select(t.c._id).where(t.c.n.contains("50%", autoescape=True))).scalars()) == [1]
            assert list(c.execute(sa.select(t.c._id).where(t.c.n.ilike("AB%"))).scalars()) == [1]
            assert list(c.execute(sa.select(t.c._id).where(t.c.n.like("%(k)s"))).scalars()) == []
            page = c.execute(sa.select(t.c._id).order_by(t.c._id).limit(2).offset(1)).scalars()
            assert list(page) == [2, 3]
            distinct = sa.func.count(sa.distinct(t.c.v)).label("d")
            rows = c.execute(sa.select(t.c.g, distinct).group_by(t.c.g).order_by(t.c.g)).fetchall()
            assert [tuple(r) for r in rows] == [("a", 2), ("b", 1)]
