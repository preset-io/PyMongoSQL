# -*- coding: utf-8 -*-
"""WHERE translation over the parse tree with SQL three-valued logic.

Every predicate yields two MongoDB filters: the documents for which it is TRUE and
those for which it is FALSE. A NULL (or missing) operand makes a comparison
UNKNOWN, which is in neither set. ``NOT p`` swaps the two sets, and De Morgan's
laws combine them for AND and OR, so ``NOT a = 1`` excludes documents where ``a``
is NULL or missing, exactly like SQL.

Predicates are read from the parse tree, never from concatenated token text: the
field is the path node on one side of the operator and the value is the literal,
parameter or value function on the other. Anything else raises
``NotSupportedError`` rather than matching different documents.
"""

import datetime
import re
from typing import Any, Dict, List, Optional, Tuple

from bson import Decimal128

from ..error import NotSupportedError
from .partiql.PartiQLParser import PartiQLParser

Filter = Dict[str, Any]
Pair = Tuple[Filter, Filter]

# Matches no document; used for predicates that can never be TRUE (or FALSE).
NOTHING: Filter = {"$expr": False}

# Stands for a bound parameter (``?``) inside a translated filter. A dict cannot be
# produced by any SQL literal, so a string literal '?' is never mistaken for it.
PARAM_KEY = "$pymongosqlParam"


def param_marker() -> Dict[str, Any]:
    return {PARAM_KEY: True}


def is_param(value: Any) -> bool:
    return isinstance(value, dict) and list(value) == [PARAM_KEY]


_LEAVES = (
    PartiQLParser.PredicateComparisonContext,
    PartiQLParser.PredicateIsContext,
    PartiQLParser.PredicateInContext,
    PartiQLParser.PredicateLikeContext,
    PartiQLParser.PredicateBetweenContext,
)
_PATH_NODES = (
    PartiQLParser.VariableIdentifierContext,
    PartiQLParser.VariableKeywordContext,
    PartiQLParser.ExprPrimaryPathContext,
)
_SWAP = {"<": ">=", ">=": "<", ">": "<=", "<=": ">"}
_MIRROR = {"<": ">", ">": "<", "<=": ">=", ">=": "<=", "=": "=", "!=": "!=", "<>": "<>"}
_MONGO = {"<": "$lt", "<=": "$lte", ">": "$gt", ">=": "$gte"}


class _Field(str):
    """A document field path (as opposed to a string value)."""


class _InexactDecimal:
    """A decimal literal that no double represents exactly (e.g. 0.1).

    SQL compares a literal in the column's type. A double field is compared with the
    nearest double and every other numeric type with the exact Decimal128, so neither
    a double 0.1 nor a Decimal128 0.1 is missed.
    """

    def __init__(self, text: str):
        self.as_double = float(text)
        self.as_decimal = Decimal128(text)

    def __neg__(self) -> "_InexactDecimal":
        negated = _InexactDecimal("0")
        negated.as_double, negated.as_decimal = -self.as_double, Decimal128(-self.as_decimal.to_decimal())
        return negated


def _decimal_literal(text: str) -> Any:
    from decimal import Decimal

    exact = Decimal(text)
    as_double = float(text)
    return as_double if Decimal(as_double) == exact else _InexactDecimal(text)


def _coerce(value: Any, as_double: bool) -> Any:
    if isinstance(value, _InexactDecimal):
        return value.as_double if as_double else value.as_decimal
    if isinstance(value, (list, tuple)):
        return type(value)(_coerce(v, as_double) for v in value)
    return value


def _has_inexact(value: Any) -> bool:
    if isinstance(value, (list, tuple)):
        return any(_has_inexact(v) for v in value)
    return isinstance(value, _InexactDecimal)


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


def _unwrap(ctx: Any) -> Any:
    """Descend through single-child pass-through rules (MathOp00 > ... > ExprTermBase)."""
    while True:
        children = getattr(ctx, "children", None) or []
        if len(children) == 1 and hasattr(children[0], "getRuleIndex"):
            ctx = children[0]
        else:
            return ctx


def _string_literal(token_text: str) -> str:
    return token_text[1:-1].replace("''", "'")


def _field_path(ctx: Any) -> str:
    """Dot path for a variable reference or path expression, quotes removed."""
    if isinstance(ctx, (PartiQLParser.VariableIdentifierContext, PartiQLParser.VariableKeywordContext)):
        if getattr(ctx, "qualifier", None) is not None:
            raise NotSupportedError(f"Unsupported variable reference: {ctx.getText()}")
        text = ctx.getText()
        return text[1:-1].replace('""', '"') if text.startswith('"') else text
    parts = [_field_path(_unwrap(ctx.getChild(0)))]
    for step in ctx.children[1:]:
        if isinstance(step, PartiQLParser.PathStepDotExprContext):
            key = step.key.getText()
            parts.append(key[1:-1].replace('""', '"') if key.startswith('"') else key)
        elif isinstance(step, PartiQLParser.PathStepIndexExprContext):
            key = _unwrap(step.key)
            if isinstance(key, PartiQLParser.LiteralIntegerContext):
                parts.append(key.getText())
            elif isinstance(key, PartiQLParser.LiteralStringContext):
                parts.append(_string_literal(key.getText()))
            else:
                raise NotSupportedError(f"Unsupported path step: {step.getText()}")
        else:
            raise NotSupportedError(f"Unsupported path step: {step.getText()}")
    path = ".".join(parts)
    # A quoted identifier with dots ("user.name") keeps this driver's nested-path meaning
    if any(not segment or segment.startswith("$") for segment in path.split(".")):
        raise NotSupportedError(f"Unsupported field name: {path!r}")
    return path


def operand(ctx: Any, resolver: Any = None) -> Any:
    """Evaluate one side of a predicate: a _Field, a Python value or a parameter marker.

    ``resolver`` may map a node (e.g. an aggregate call in HAVING) to a field name.
    """
    node = _unwrap(ctx)
    if resolver is not None:
        resolved = resolver(node)
        if resolved is not None:
            return _Field(resolved)
    if isinstance(node, _PATH_NODES):
        return _Field(_field_path(node))
    if isinstance(node, PartiQLParser.ParameterContext):
        return param_marker()
    if isinstance(node, PartiQLParser.LiteralStringContext):
        return _string_literal(node.getText())
    if isinstance(node, PartiQLParser.LiteralIntegerContext):
        return int(node.getText())
    if isinstance(node, PartiQLParser.LiteralDecimalContext):
        return _decimal_literal(node.getText())
    if isinstance(node, PartiQLParser.LiteralTrueContext):
        return True
    if isinstance(node, PartiQLParser.LiteralFalseContext):
        return False
    if isinstance(node, (PartiQLParser.LiteralNullContext, PartiQLParser.LiteralMissingContext)):
        return None
    if isinstance(node, PartiQLParser.LiteralDateContext):
        return datetime.datetime.fromisoformat(_string_literal(node.LITERAL_STRING().getText()))
    if isinstance(node, PartiQLParser.ValueExprContext) and node.sign is not None:
        value = operand(node.rhs)
        if isinstance(value, bool) or not isinstance(value, (int, float, _InexactDecimal)):
            raise NotSupportedError(f"Unsupported signed expression: {node.getText()}")
        return value if node.sign.text == "+" else -value
    if isinstance(node, PartiQLParser.MathOp00Context) and node.op is not None and node.op.text == "||":
        left, right = operand(node.lhs), operand(node.rhs)
        if (
            not (isinstance(left, str) and isinstance(right, str))
            or isinstance(left, _Field)
            or isinstance(right, _Field)
        ):
            raise NotSupportedError(f"Only string literals can be concatenated: {node.getText()}")
        return left + right
    if isinstance(node, PartiQLParser.FunctionCallContext):
        from .value_function_registry import get_default_registry

        name = node.functionName().getText()
        registry = get_default_registry()
        if registry.has_function(name):
            args = [_coerce(operand(arg), as_double=True) for arg in node.expr()]
            if any(isinstance(a, _Field) or is_param(a) for a in args):
                raise NotSupportedError(f"Value functions take literal arguments: {node.getText()}")
            return registry.execute(name, args)
    raise NotSupportedError(f"Unsupported WHERE operand: {node.getText()}")


def _like_regex(pattern: str, escape: Optional[str]) -> Tuple[str, bool]:
    """Regex for a LIKE pattern and whether it needs DOTALL.

    ``escape`` makes the next character literal. A leading or trailing ``%`` leaves
    that end unanchored; a wildcard anywhere else must also match newlines.
    """
    tokens, i = [], 0
    while i < len(pattern):
        char = pattern[i]
        if escape is not None and char == escape:
            if i + 1 >= len(pattern):
                raise NotSupportedError("LIKE pattern ends with its escape character")
            tokens.append(re.escape(pattern[i + 1]))
            i += 2
            continue
        tokens.append(".*" if char == "%" else "." if char == "_" else re.escape(char))
        i += 1
    body = "".join(tokens)
    inner = tokens[1 if tokens[:1] == [".*"] else 0 : len(tokens) - (1 if tokens[-1:] == [".*"] else 0)]
    needs_dotall = any(t in (".*", ".") for t in inner)
    regex = ("" if body.startswith(".*") else "^") + body + ("" if body.endswith(".*") else "$")
    return regex, needs_dotall


def leaf_filters(field: str, operator: str, value: Any, escape: Optional[str] = None) -> Pair:
    """TRUE and FALSE filters for one predicate on ``field``."""
    if _has_inexact(value):
        double = {field: {"$type": "double"}}
        other = {field: {"$not": {"$type": "double"}}}
        t1, f1 = leaf_filters(field, operator, _coerce(value, True), escape)
        t2, f2 = leaf_filters(field, operator, _coerce(value, False), escape)
        return (
            {"$or": [{"$and": [double, t1]}, {"$and": [other, t2]}]},
            {"$or": [{"$and": [double, f1]}, {"$and": [other, f2]}]},
        )
    op = operator.upper()
    if op == "IS NULL":
        return {field: {"$eq": None}}, {field: {"$ne": None}}
    if op == "IS NOT NULL":
        return {field: {"$ne": None}}, {field: {"$eq": None}}
    if op == "IS MISSING":
        return {field: {"$exists": False}}, {field: {"$exists": True}}
    if op == "IS NOT MISSING":
        return {field: {"$exists": True}}, {field: {"$exists": False}}
    if op in ("IN", "NOT IN"):
        values = value if isinstance(value, list) else [value]
        present = [v for v in values if v is not None]
        true = {field: {"$in": present}}
        # x NOT IN (..., NULL) is never TRUE; x IN (..., NULL) is never FALSE
        false = NOTHING if None in values else {field: {"$nin": present + [None]}}
        return (true, false) if op == "IN" else (false, true)
    if op in ("LIKE", "NOT LIKE"):
        if is_param(value) or not isinstance(value, str):
            # The pattern becomes a regex while parsing, before parameters are bound
            raise NotSupportedError("LIKE needs a literal pattern, not a bound parameter")
        pattern, dotall = _like_regex(value, escape)
        regex = {"$regex": pattern, "$options": "s"} if dotall else {"$regex": pattern}
        true = {field: regex}
        false = {"$and": [{field: {"$not": regex}}, {field: {"$ne": None}}]}
        return (true, false) if op == "LIKE" else (false, true)
    if op in ("BETWEEN", "NOT BETWEEN"):
        low, high = value
        true = {"$and": [{field: {"$gte": low}}, {field: {"$lte": high}}]}
        false = {"$or": [{field: {"$lt": low}}, {field: {"$gt": high}}]}
        return (true, false) if op == "BETWEEN" else (false, true)
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


def _value(ctx: Any, resolver: Any = None) -> Any:
    value = operand(ctx, resolver)
    if isinstance(value, _Field):
        raise NotSupportedError(f"Comparing two fields is not supported: {ctx.getText()}")
    return value


class WhereTreeBuilder:
    """Build a MongoDB filter for a WHERE (or HAVING) expression from its parse tree."""

    def __init__(self, resolver: Any = None):
        self._resolver = resolver

    def build(self, ctx: Any) -> Filter:
        return self._pair(ctx)[0]

    def _pair(self, ctx: Any, substitute: Optional[Tuple[Any, Pair]] = None) -> Pair:
        if substitute is not None and ctx is substitute[0]:
            return substitute[1]
        if isinstance(ctx, PartiQLParser.NotContext):
            true, false = self._pair(ctx.rhs, substitute)
            return false, true
        if isinstance(ctx, (PartiQLParser.AndContext, PartiQLParser.OrContext)):
            is_and = isinstance(ctx, PartiQLParser.AndContext)
            (t1, f1), (t2, f2) = self._pair(ctx.lhs, substitute), self._pair(ctx.rhs, substitute)
            true_key, false_key = ("$and", "$or") if is_and else ("$or", "$and")
            chain = type(ctx)
            return (
                _all([(ctx.lhs, t1), (ctx.rhs, t2)], true_key, chain),
                _all([(ctx.lhs, f1), (ctx.rhs, f2)], false_key, chain),
            )
        if isinstance(ctx, PartiQLParser.ExprTermWrappedQueryContext):
            return self._pair(ctx.expr(), substitute)
        if isinstance(ctx, PartiQLParser.PredicateLikeContext):
            return self._like(ctx)
        if isinstance(ctx, _LEAVES):
            return self._leaf(ctx)
        children = [c for c in getattr(ctx, "children", None) or [] if hasattr(c, "getRuleIndex")]
        if len(children) == 1 and len(ctx.children) == 1:
            return self._pair(children[0], substitute)
        if isinstance(ctx, _PATH_NODES):
            # A bare boolean field: WHERE flag / WHERE NOT flag
            field = str(operand(ctx, self._resolver))
            return {field: True}, {field: False}
        if isinstance(ctx, (PartiQLParser.LiteralTrueContext, PartiQLParser.LiteralFalseContext)):
            everything: Filter = {}
            return (everything, NOTHING) if isinstance(ctx, PartiQLParser.LiteralTrueContext) else (NOTHING, everything)
        raise NotSupportedError(f"Unsupported WHERE expression: {ctx.getText()}")

    def _field_and_value(self, lhs: Any, rhs: Any, text: str) -> Tuple[str, Any, bool]:
        """(field, value, mirrored) for ``lhs op rhs`` with the field on either side."""
        left, right = operand(lhs, self._resolver), operand(rhs, self._resolver)
        if isinstance(left, _Field) and not isinstance(right, _Field):
            return str(left), right, False
        if isinstance(right, _Field) and not isinstance(left, _Field):
            return str(right), left, True
        raise NotSupportedError(f"A predicate needs exactly one field and one value: {text}")

    def _leaf(self, ctx: Any) -> Pair:
        text = ctx.getText()
        if isinstance(ctx, PartiQLParser.PredicateComparisonContext):
            field, value, mirrored = self._field_and_value(ctx.lhs, ctx.rhs, text)
            op = ctx.op.text
            return leaf_filters(field, _MIRROR[op] if mirrored else op, value)
        negated = ctx.NOT() is not None
        field = operand(ctx.lhs, self._resolver)
        if not isinstance(field, _Field):
            raise NotSupportedError(f"The left side must be a field: {text}")
        if isinstance(ctx, PartiQLParser.PredicateIsContext):
            kind = ctx.type_().getText().upper()
            if kind not in ("NULL", "MISSING"):
                raise NotSupportedError(f"Unsupported IS type test: {text}")
            return leaf_filters(field, f"IS {'NOT ' if negated else ''}{kind}", None)
        if isinstance(ctx, PartiQLParser.PredicateInContext):
            target = _unwrap(ctx.rhs) if ctx.rhs is not None else None
            if isinstance(target, PartiQLParser.ValueListContext):
                values = [_value(item, self._resolver) for item in target.expr()]
            elif ctx.expr() is not None and ctx.rhs is None:
                values = [_value(ctx.expr(), self._resolver)]  # IN (single value)
            else:
                raise NotSupportedError(f"Unsupported IN list: {text}")
            return leaf_filters(field, "NOT IN" if negated else "IN", values)
        if isinstance(ctx, PartiQLParser.PredicateBetweenContext):
            bounds = (_value(ctx.lower, self._resolver), _value(ctx.upper, self._resolver))
            return leaf_filters(field, "NOT BETWEEN" if negated else "BETWEEN", bounds)
        raise NotSupportedError(f"Unsupported predicate: {text}")

    def _like(self, ctx: Any) -> Pair:
        field = operand(ctx.lhs, self._resolver)
        if not isinstance(field, _Field):
            raise NotSupportedError(f"The left side of LIKE must be a field: {ctx.getText()}")
        pattern = _value(ctx.rhs, self._resolver)
        op = "NOT LIKE" if ctx.NOT() is not None else "LIKE"
        if ctx.escape is None:
            return leaf_filters(field, op, pattern)
        # The grammar lets ESCAPE take a whole expression, so "a LIKE p ESCAPE '/' AND b = 1"
        # parses the AND chain as the escape. Its leftmost operand is the escape character;
        # the LIKE predicate takes that operand's place in the chain.
        leftmost = _unwrap(ctx.escape)
        while isinstance(leftmost, (PartiQLParser.AndContext, PartiQLParser.OrContext)):
            leftmost = _unwrap(leftmost.lhs)
        escape = _value(leftmost, self._resolver)
        if not isinstance(escape, str) or len(escape) != 1:
            raise NotSupportedError(f"LIKE ESCAPE must be a single character: {ctx.getText()}")
        like = leaf_filters(field, op, pattern, escape)
        if leftmost is _unwrap(ctx.escape):
            return like
        return self._pair(ctx.escape, substitute=(leftmost, like))
