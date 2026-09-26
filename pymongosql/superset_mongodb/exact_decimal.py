# -*- coding: utf-8 -*-
"""Exact decimal evaluation for the superset-mode SQLite stage.

SQLite has no decimal type: a Decimal128 value stored there becomes a double (15-17
significant digits) or text (which sorts and compares as a string). A column whose
values include decimals is therefore stored twice: as REAL in the query table (so
SQLite can still filter and sort it approximately) and as exact decimal text in a
side table keyed by rowid. Before a query runs, it is rewritten so that projections
of such a column, SUM/AVG/MIN/MAX over it, GROUP BY, ORDER BY and comparisons with a
numeric literal are evaluated from the exact text with Decimal arithmetic in the
precision of MongoDB's Decimal128 (34 significant digits). A query that uses such a
column in any other way (arithmetic, functions, DISTINCT aggregates) raises instead
of silently computing with doubles.
"""

from decimal import ROUND_HALF_EVEN, Context, Decimal
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..error import NotSupportedError

SHADOW_PREFIX = "__pymongosql_exact__"
# Exact enough for any sum of Decimal128 values; results are then rounded like MongoDB's
# own Decimal128 arithmetic.
_SUM_CONTEXT = Context(prec=13000, rounding=ROUND_HALF_EVEN, Emin=-99999, Emax=99999)
_DECIMAL128 = Context(prec=34, rounding=ROUND_HALF_EVEN, Emin=-6143, Emax=6144)


def _decimal(value: Optional[str]) -> Optional[Decimal]:
    return None if value is None else Decimal(value)


def _round(value: Decimal) -> str:
    return str(_DECIMAL128.plus(value))


class _Collect:
    def __init__(self) -> None:
        self.values: List[Decimal] = []

    def step(self, value: Optional[str]) -> None:
        if value is not None:
            self.values.append(Decimal(value))

    def total(self) -> Decimal:
        total = Decimal(0)
        for value in self.values:
            total = _SUM_CONTEXT.add(total, value)
        return total


class _Sum(_Collect):
    def finalize(self) -> Optional[str]:
        return _round(self.total()) if self.values else None


class _Avg(_Collect):
    def finalize(self) -> Optional[str]:
        if not self.values:
            return None
        return str(_DECIMAL128.divide(self.total(), Decimal(len(self.values))))


class _Min(_Collect):
    def finalize(self) -> Optional[str]:
        return str(min(self.values)) if self.values else None


class _Max(_Collect):
    def finalize(self) -> Optional[str]:
        return str(max(self.values)) if self.values else None


def sort_key(value: Optional[str]) -> Optional[str]:
    """Text whose byte order is the numeric order of the decimal ``value``."""
    number = _decimal(value)
    if number is None:
        return None
    if number == 0:
        return "1"
    digits = "".join(map(str, number.normalize().as_tuple().digits))
    exponent = number.adjusted()
    if number > 0:
        return "2%06d%s" % (exponent + 100000, digits)
    # Negative: larger magnitude first; "~" makes a prefix (-1) sort after -1.2
    complement = "".join(str(9 - int(d)) for d in digits)
    return "0%06d%s~" % (100000 - exponent, complement)


def compare(value: Optional[str], other: Optional[str]) -> Optional[int]:
    left, right = _decimal(value), _decimal(other)
    if left is None or right is None:
        return None
    return (left > right) - (left < right)


AGGREGATES = {"SUM": ("__pymongosql_exact_sum", _Sum), "AVG": ("__pymongosql_exact_avg", _Avg)}
AGGREGATES.update({"MIN": ("__pymongosql_exact_min", _Min), "MAX": ("__pymongosql_exact_max", _Max)})
FUNCTIONS = {"__pymongosql_exact_key": sort_key, "__pymongosql_exact_cmp": compare}


def register(connection: Any) -> None:
    for name, aggregate in AGGREGATES.values():
        connection.create_aggregate(name, 1, aggregate)
    for name, function in FUNCTIONS.items():
        connection.create_function(name, 2 if name.endswith("cmp") else 1, function, deterministic=True)


def rewrite(sql: str, table: str, exact_columns: Sequence[str], columns: Sequence[str] = ()) -> Tuple[str, List[int]]:
    """Rewrite ``sql`` to evaluate the exact columns of ``table`` exactly.

    Returns the SQL and the positions of the outer projections that return exact
    decimal text. ``columns`` (all columns of ``table``, in order) expands ``SELECT *``.
    Raises NotSupportedError when an exact column is used in a way that
    cannot be evaluated exactly.
    """
    try:
        import sqlglot
        from sqlglot import exp
    except ImportError as e:  # pragma: no cover - sqlglot ships with Superset
        raise NotSupportedError("Exact decimal evaluation needs the sqlglot package") from e

    exact = set(exact_columns)
    try:
        tree = sqlglot.parse_one(sql, read="sqlite")
    except sqlglot.errors.ParseError as e:
        raise NotSupportedError(f"Cannot evaluate decimal columns exactly in: {sql}") from e
    if not isinstance(tree, exp.Select):
        raise NotSupportedError(f"Cannot evaluate decimal columns exactly in: {sql}")

    comparisons = (exp.EQ, exp.NEQ, exp.GT, exp.GTE, exp.LT, exp.LTE)
    mirrored = {exp.GT: exp.LT, exp.GTE: exp.LTE, exp.LT: exp.GT, exp.LTE: exp.GTE}
    aggregates = {exp.Sum: "SUM", exp.Avg: "AVG", exp.Min: "MIN", exp.Max: "MAX"}

    def reads_table(select: Any) -> bool:
        from_ = select.args.get("from") or select.args.get("from_")
        return (
            from_ is not None
            and isinstance(from_.this, exp.Table)
            and not from_.this.db
            and from_.this.name == table
            and not select.args.get("joins")
        )

    def numeric_literal(node: Any) -> Optional[str]:
        if isinstance(node, exp.Literal) and not node.is_string:
            return str(Decimal(node.this))
        if isinstance(node, exp.Neg):
            inner = numeric_literal(node.this)
            return None if inner is None else str(-Decimal(inner))
        return None

    def shadow(name: str) -> Any:
        return exp.column(SHADOW_PREFIX + name, quoted=True)

    def call(name: str, *args: Any) -> Any:
        return exp.Anonymous(this=name, expressions=list(args))

    def expand_star(select: Any) -> None:
        expanded = []
        for projection in select.expressions:
            star = isinstance(projection, exp.Star) or (
                isinstance(projection, exp.Column) and isinstance(projection.this, exp.Star)
            )
            if star and columns:
                expanded.extend(exp.column(name, quoted=True) for name in columns)
            else:
                expanded.append(projection)
        select.set("expressions", expanded)

    def process(select: Any, outer: bool) -> List[int]:
        expand_star(select)

        def column(node: Any) -> Optional[str]:
            return node.name if isinstance(node, exp.Column) and node.name in exact else None

        def value(node: Any) -> Optional[Any]:
            name = column(node)
            if name is not None:
                return shadow(name)
            if type(node) in aggregates and not isinstance(node.this, exp.Distinct):
                name = column(node.this)
                if name is not None:
                    return call(AGGREGATES[aggregates[type(node)]][0], shadow(name))
            return None

        by_name: Dict[str, Any] = {}
        by_position: Dict[int, Any] = {}
        for index, projection in enumerate(select.expressions):
            inner = projection.this if isinstance(projection, exp.Alias) else projection
            by_position[index] = inner
            by_name.setdefault(projection.alias_or_name, inner)

        def referenced(node: Any) -> Any:
            if isinstance(node, exp.Literal) and node.is_int:
                return by_position.get(int(node.this) - 1, node)
            if isinstance(node, exp.Column) and not node.table:
                # SQLite resolves a bare name to an output alias before a column
                return by_name.get(node.name, node)
            return node

        for key in ("where", "having"):
            clause = select.args.get(key)
            if clause is None:
                continue
            for comparison in list(clause.find_all(*comparisons)):
                if comparison.find_ancestor(exp.Select) is not select:
                    continue
                left, right, kind = comparison.this, comparison.expression, type(comparison)
                if numeric_literal(left) is not None:
                    left, right, kind = right, left, mirrored.get(kind, kind)
                literal = numeric_literal(right)
                exact_left = value(referenced(left)) if literal is not None else None
                if exact_left is not None:
                    comparison.replace(
                        kind(
                            this=call("__pymongosql_exact_cmp", exact_left, exp.Literal.string(literal)),
                            expression=exp.Literal.number(0),
                        )
                    )
        group = select.args.get("group")
        if group is not None:
            group.set(
                "expressions",
                [shadow(column(referenced(k))) if column(referenced(k)) else k for k in group.expressions],
            )
        order = select.args.get("order")
        if order is not None:
            for ordered in order.expressions:
                exact_order = value(referenced(ordered.this))
                if exact_order is not None:
                    ordered.set("this", call("__pymongosql_exact_key", exact_order))
        positions: List[int] = []
        projections = []
        for index, projection in enumerate(select.expressions):
            inner = projection.this if isinstance(projection, exp.Alias) else projection
            exact_projection = value(inner)
            if exact_projection is None:
                projections.append(projection)
                continue
            if outer:
                positions.append(index)
            name = projection.alias_or_name
            projections.append(exp.alias_(exact_projection, name, quoted=True))
        select.set("expressions", projections)
        return positions

    selects = [s for s in tree.find_all(exp.Select) if reads_table(s)]
    positions: List[int] = []
    for select in selects:
        found = process(select, outer=select is tree)
        if select is tree:
            positions = found
    stars = [s for s in tree.find_all(exp.Star) if not isinstance(s.parent, exp.Count)]
    if stars:
        # A * over a derived table would return the exact text untyped
        raise NotSupportedError("SELECT * over a derived table with decimal columns: name the columns")
    # Any remaining reference to an exact column outside COUNT or IS NULL would be
    # computed with doubles
    for node in tree.find_all(exp.Column):
        if node.name in exact:
            allowed = node.find_ancestor(exp.Count, exp.Is)
            if allowed is None:
                raise NotSupportedError(
                    f"Decimal column {node.name!r} is used in an expression that cannot be evaluated exactly"
                )
    columns = ", ".join(f'x."{c}" AS "{SHADOW_PREFIX}{c}"' for c in exact_columns)
    source = f'(SELECT q.*, {columns} FROM "{table}" AS q LEFT JOIN "{table}_exact" AS x ON x.rid = q.rowid)'
    for node in list(tree.find_all(exp.Table)):
        if node.name == table and not node.db:
            alias = node.alias or table
            node.replace(
                exp.Subquery(
                    this=sqlglot.parse_one(source[1:-1], read="sqlite"),
                    alias=exp.TableAlias(this=exp.to_identifier(alias, quoted=True)),
                )
            )
    return tree.sql(dialect="sqlite"), positions
