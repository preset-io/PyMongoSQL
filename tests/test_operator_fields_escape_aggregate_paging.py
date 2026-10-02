# -*- coding: utf-8 -*-
"""Quoted operator names in WHERE.

- A field name starting with ``$`` is rejected whether or not it is quoted, so
  ``"$where"`` or ``"$expr"`` can never reach MongoDB as a query operator.
"""

import pytest

from pymongosql.error import Error
from pymongosql.sql.parser import SQLParser


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
