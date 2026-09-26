# -*- coding: utf-8 -*-
"""WHERE translation over the parse tree with SQL three-valued logic.

Every predicate yields two MongoDB filters: the documents for which it is TRUE and
those for which it is FALSE. A NULL (or missing) operand makes a comparison
UNKNOWN, which is in neither set. ``NOT p`` swaps the two sets, and De Morgan's
laws combine them for AND and OR, so ``NOT a = 1`` excludes documents where ``a``
is NULL or missing, exactly like SQL.
"""

import re
from typing import Any, Dict, List, Tuple

from ..error import NotSupportedError
from .partiql.PartiQLParser import PartiQLParser

Filter = Dict[str, Any]
Pair = Tuple[Filter, Filter]

# Matches no document; used for predicates that can never be TRUE (or FALSE).
NOTHING: Filter = {"$expr": False}

_LEAVES = (
    PartiQLParser.PredicateComparisonContext,
    PartiQLParser.PredicateIsContext,
    PartiQLParser.PredicateInContext,
    PartiQLParser.PredicateLikeContext,
    PartiQLParser.PredicateBetweenContext,
)
_FIELD_PATH = re.compile(r'^(?:"[^"]+"|[A-Za-z_$][\w$]*)(?:\.(?:"[^"]+"|[A-Za-z_$][\w$]*|\d+))*$')
_FIELD = re.compile(r"^[A-Za-z_$][\w$]*(?:\.[\w$]+)*$")
_SWAP = {"<": ">=", ">=": "<", ">": "<=", "<=": ">"}
_MONGO = {"<": "$lt", "<=": "$lte", ">": "$gt", ">=": "$gte"}


def _all(parts: List[Tuple[Any, Filter]], key: str, chain: type) -> Filter:
    """Combine filters under ``key``, flattening only an unparenthesized chain of the same operator."""
    items: List[Filter] = []
    for ctx, f in parts:
        items.extend(f[key] if isinstance(ctx, chain) and list(f) == [key] else [f])
    return {key: items}


def contains_not(ctx: Any) -> bool:
    """Whether the expression has a boolean NOT (not a NOT IN / NOT LIKE predicate)."""
    if isinstance(ctx, PartiQLParser.NotContext):
        return True
    return any(contains_not(child) for child in getattr(ctx, "children", None) or [])


def leaf_filters(field: str, operator: str, value: Any) -> Pair:
    """TRUE and FALSE filters for one predicate on ``field``."""
    op = operator.upper()
    if op == "IS NULL":
        return {field: {"$eq": None}}, {field: {"$ne": None}}
    if op == "IS NOT NULL":
        return {field: {"$ne": None}}, {field: {"$eq": None}}
    if op in ("IN", "NOT IN"):
        values = value if isinstance(value, list) else [value]
        present = [v for v in values if v is not None]
        true = {field: {"$in": present}}
        # x NOT IN (..., NULL) is never TRUE; x IN (..., NULL) is never FALSE
        false = NOTHING if None in values else {field: {"$nin": present + [None]}}
        return (true, false) if op == "IN" else (false, true)
    if op in ("LIKE", "NOT LIKE"):
        from .handler import ComparisonExpressionHandler

        if value == "?" or not isinstance(value, str):
            # The pattern is translated to a regex while parsing, before parameters are bound
            raise NotSupportedError("LIKE needs a literal pattern, not a bound parameter")

        pattern = ComparisonExpressionHandler._like_to_regex(value)
        pattern = ("" if pattern.startswith(".*") else "^") + pattern + ("" if pattern.endswith(".*") else "$")
        true = {field: {"$regex": pattern}}
        false = {"$and": [{field: {"$not": {"$regex": pattern}}}, {field: {"$ne": None}}]}
        return (true, false) if op == "LIKE" else (false, true)
    if op == "BETWEEN":
        low, high = value
        true = {"$and": [{field: {"$gte": low}}, {field: {"$lte": high}}]}
        return true, {"$or": [{field: {"$lt": low}}, {field: {"$gt": high}}]}
    if value is None:
        # This dialect has always read "= NULL" / "<> NULL" as IS NULL / IS NOT NULL;
        # any other comparison with NULL is UNKNOWN.
        if op == "=":
            return {field: None}, {field: {"$ne": None}}
        if op in ("!=", "<>"):
            return {field: {"$ne": None}}, {field: None}
        return NOTHING, NOTHING
    if op == "=":
        return {field: value}, {field: {"$nin": [value, None]}}
    if op in ("!=", "<>"):
        return {field: {"$nin": [value, None]}}, {field: value}
    if op in _MONGO:
        return {field: {_MONGO[op]: value}}, {field: {_MONGO[_SWAP[op]]: value}}
    raise NotSupportedError(f"Unsupported predicate operator: {operator}")


class WhereTreeBuilder:
    """Build a MongoDB filter for a WHERE expression from its parse tree."""

    def build(self, ctx: Any) -> Filter:
        return self._pair(ctx)[0]

    def _pair(self, ctx: Any) -> Pair:
        if isinstance(ctx, PartiQLParser.NotContext):
            true, false = self._pair(ctx.rhs)
            return false, true
        if isinstance(ctx, (PartiQLParser.AndContext, PartiQLParser.OrContext)):
            is_and = isinstance(ctx, PartiQLParser.AndContext)
            (t1, f1), (t2, f2) = self._pair(ctx.lhs), self._pair(ctx.rhs)
            true_key, false_key = ("$and", "$or") if is_and else ("$or", "$and")
            chain = type(ctx)
            return (
                _all([(ctx.lhs, t1), (ctx.rhs, t2)], true_key, chain),
                _all([(ctx.lhs, f1), (ctx.rhs, f2)], false_key, chain),
            )
        if isinstance(ctx, PartiQLParser.ExprTermWrappedQueryContext):
            return self._pair(ctx.expr())
        if isinstance(ctx, _LEAVES):
            return self._leaf(ctx)
        children = [c for c in getattr(ctx, "children", None) or [] if hasattr(c, "getRuleIndex")]
        if len(children) == 1 and len(ctx.children) == 1:
            return self._pair(children[0])
        text = ctx.getText()
        if _FIELD_PATH.match(text) and text.upper() not in ("TRUE", "FALSE", "NULL"):
            # A bare boolean field: WHERE flag / WHERE NOT flag
            from .handler import ContextUtilsMixin

            field = ContextUtilsMixin.normalize_field_path(text)
            return {field: True}, {field: False}
        raise NotSupportedError(f"Unsupported WHERE expression: {text}")

    @staticmethod
    def _leaf(ctx: Any) -> Pair:
        from .handler import ComparisonExpressionHandler

        handler = ComparisonExpressionHandler()
        field = handler._extract_field_name(ctx)
        operator = handler._extract_operator(ctx)
        value = handler._extract_value(ctx)
        if not _FIELD.match(field):
            raise NotSupportedError(f"Unsupported WHERE predicate: {ctx.getText()}")
        return leaf_filters(field, operator, value)
