# -*- coding: utf-8 -*-
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .handler import BaseHandler, ContextUtilsMixin
from .partiql.PartiQLParser import PartiQLParser

_logger = logging.getLogger(__name__)


@dataclass
class QueryParseResult:
    """Result container for query (SELECT) expression parsing and visitor state management"""

    # Core parsing fields
    filter_conditions: Dict[str, Any] = field(default_factory=dict)  # Unified filter field for all MongoDB conditions
    has_errors: bool = False
    error_message: Optional[str] = None

    # Visitor parsing state fields
    collection: Optional[str] = None
    projection: Dict[str, Any] = field(default_factory=dict)
    column_aliases: Dict[str, str] = field(default_factory=dict)  # Maps field_name -> alias
    sort_fields: List[Dict[str, int]] = field(default_factory=list)
    limit_value: Optional[int] = None
    offset_value: Optional[int] = None

    # Aggregate pipeline support
    is_aggregate_query: bool = False  # Flag indicating this is an aggregate() call
    aggregate_pipeline: Optional[str] = None  # JSON string representation of pipeline
    aggregate_options: Optional[str] = None  # JSON string representation of options

    # SQL aggregate functions detected in SELECT (COUNT, SUM, AVG, MIN, MAX)
    aggregate_functions: List[Dict[str, Any]] = field(default_factory=list)
    # SELECT items in order: {"field": name, "alias": alias} or {"aggregate": index}
    select_items: List[Dict[str, Any]] = field(default_factory=list)
    # GROUP BY field paths (or the text of a computed expression, see ``computed``)
    group_by: List[str] = field(default_factory=list)
    # Computed expressions by their SQL text: {"unit": ..., "field": ...} for DATE_TRUNC
    computed: Dict[str, Dict[str, str]] = field(default_factory=dict)
    # Clauses that are parsed but cannot be translated faithfully
    unsupported_clauses: List[str] = field(default_factory=list)
    # FROM alias (FROM users AS u / FROM users u)
    collection_alias: Optional[str] = None
    # HAVING expression (parse-tree node), translated after grouping
    having: Any = None

    # Subquery info (for wrapped subqueries, e.g., Superset outering)
    subquery_plan: Optional[Any] = None
    subquery_alias: Optional[str] = None

    # Factory methods for different use cases
    @classmethod
    def for_visitor(cls) -> "QueryParseResult":
        """Create QueryParseResult for visitor parsing"""
        return cls()

    def merge_expression(self, other: "QueryParseResult") -> "QueryParseResult":
        """Merge expression results from another QueryParseResult"""
        if other.has_errors:
            self.has_errors = True
            self.error_message = other.error_message

        # Merge filter conditions intelligently
        if other.filter_conditions:
            if not self.filter_conditions:
                self.filter_conditions = other.filter_conditions
            else:
                # If both have filters, combine them with $and
                self.filter_conditions = {"$and": [self.filter_conditions, other.filter_conditions]}

        return self

    # Backward compatibility properties
    @property
    def mongo_filter(self) -> Dict[str, Any]:
        """Backward compatibility property for mongo_filter"""
        return self.filter_conditions

    @mongo_filter.setter
    def mongo_filter(self, value: Dict[str, Any]):
        """Backward compatibility setter for mongo_filter"""
        self.filter_conditions = value


class EnhancedWhereHandler(ContextUtilsMixin):
    """Enhanced WHERE clause handler using expression handlers"""

    def handle(self, ctx: PartiQLParser.WhereClauseSelectContext) -> Dict[str, Any]:
        """Handle WHERE clause with proper expression parsing"""
        if not hasattr(ctx, "exprSelect") or not ctx.exprSelect():
            _logger.debug("No expression found in WHERE clause")
            return {}

        expression_ctx = ctx.exprSelect()
        # Local import to avoid circular dependency between query_handler and handler
        from .handler import HandlerFactory

        handler = HandlerFactory.get_expression_handler(expression_ctx)

        if handler:
            _logger.debug(
                f"Using {type(handler).__name__} for WHERE clause",
                extra={"context_text": self.get_context_text(expression_ctx)[:100]},
            )
            result = handler.handle_expression(expression_ctx)
            if result.has_errors:
                _logger.warning(
                    "Expression parsing error, falling back to text search",
                    extra={"error": result.error_message},
                )
                # Fallback to text-based filter
                return {"$text": {"$search": self.get_context_text(expression_ctx)}}
            return result.filter_conditions
        else:
            # Fallback to simple text-based search
            _logger.debug(
                "No suitable expression handler found, using text search",
                extra={"context_text": self.get_context_text(expression_ctx)[:100]},
            )
            return {"$text": {"$search": self.get_context_text(expression_ctx)}}


class SelectHandler(BaseHandler, ContextUtilsMixin):
    """Handles SELECT statement parsing"""

    # Pattern to detect SQL aggregate functions: COUNT(*), SUM(field), AVG(field), etc.
    _AGGREGATE_PATTERN = re.compile(
        r"^(COUNT|SUM|AVG|MIN|MAX)\s*\(\s*(\*|\w+(?:\.\w+)*)\s*\)$",
        re.IGNORECASE,
    )

    def can_handle(self, ctx: Any) -> bool:
        """Check if this is a select context"""
        return hasattr(ctx, "projectionItems")

    def handle_visitor(self, ctx: PartiQLParser.SelectItemsContext, parse_result: "QueryParseResult") -> Any:
        projection = {}
        column_aliases = {}

        if hasattr(ctx, "projectionItems") and ctx.projectionItems():
            for item in ctx.projectionItems().projectionItem():
                field_name, alias = self._extract_field_and_alias(item)
                kind, detail = self._classify_item(item)

                if kind == "aggregate":
                    func_name, func_arg, distinct = detail
                    parse_result.select_items.append({"aggregate": len(parse_result.aggregate_functions)})
                    parse_result.aggregate_functions.append(
                        {
                            "function": func_name,
                            "argument": func_arg,
                            "distinct": distinct,
                            "alias": alias or field_name,
                            "expression": field_name,
                        }
                    )
                    continue
                if kind == "computed":
                    parse_result.computed[field_name] = detail
                    parse_result.select_items.append({"computed": field_name, "alias": alias})
                    continue
                if kind == "unsupported":
                    # e.g. a + 1 or lower(a): projecting it as a field would silently read NULL
                    parse_result.unsupported_clauses.append(f"SELECT {detail}")
                    continue

                field_name = detail
                parse_result.select_items.append({"field": field_name, "alias": alias})
                # Use MongoDB standard projection format: {field: 1} to include field
                projection[field_name] = 1
                # Store alias if present
                if alias:
                    column_aliases[field_name] = alias

        parse_result.projection = projection
        parse_result.column_aliases = column_aliases
        return projection

    _AGGREGATES = ("COUNT", "SUM", "AVG", "MIN", "MAX")

    @staticmethod
    def date_trunc(node: Any) -> Optional[Dict[str, str]]:
        """{"unit", "field"} for DATE_TRUNC('<unit>', <field>), else None."""
        from ..superset_mongodb.time_grain import UNITS
        from .where_tree import _PATH_NODES, _field_path, _string_literal, _unwrap

        node = _unwrap(node)
        if not isinstance(node, PartiQLParser.FunctionCallContext):
            return None
        if node.functionName().getText().lower() != "date_trunc" or len(node.expr()) != 2:
            return None
        unit, target = _unwrap(node.expr()[0]), _unwrap(node.expr()[1])
        if not isinstance(unit, PartiQLParser.LiteralStringContext) or not isinstance(target, _PATH_NODES):
            raise ValueError(f"DATE_TRUNC needs a unit literal and a field: {node.getText()}")
        name = _string_literal(unit.getText()).lower()
        if name not in UNITS:
            raise ValueError(f"Unsupported DATE_TRUNC unit {name!r}; use one of {', '.join(UNITS)}")
        return {"unit": name, "field": _field_path(target)}

    @staticmethod
    def _classify_item(item) -> Tuple[str, Any]:
        """("field", path) | ("aggregate", (function, argument, distinct)) | ("unsupported", text)."""
        from .where_tree import _PATH_NODES, _field_path, _unwrap

        # A projection item's first child is its expression; other nodes are classified as-is
        is_item = isinstance(item, PartiQLParser.ProjectionItemContext)
        expr = item.children[0] if is_item and getattr(item, "children", None) else item
        if not hasattr(expr, "getRuleIndex"):
            return "unsupported", str(expr)
        node = _unwrap(expr)
        try:
            if isinstance(node, PartiQLParser.CountAllContext):
                return "aggregate", ("COUNT", "*", False)
            if isinstance(node, PartiQLParser.AggregateBaseContext):
                func = node.func.text.upper()
                quantifier = node.setQuantifierStrategy()
                argument = _unwrap(node.expr())
                if func not in SelectHandler._AGGREGATES or not isinstance(argument, _PATH_NODES):
                    return "unsupported", node.getText()
                distinct = quantifier is not None and quantifier.getText().upper() == "DISTINCT"
                return "aggregate", (func, _field_path(argument), distinct)
            if isinstance(node, _PATH_NODES):
                return "field", _field_path(node)
            truncated = SelectHandler.date_trunc(node)
            if truncated is not None:
                return "computed", truncated
        except Exception as e:
            return "unsupported", f"{node.getText()} ({e})"
        return "unsupported", node.getText()

    def _extract_field_and_alias(self, item) -> Tuple[str, Optional[str]]:
        """Extract field name and alias from projection item context with nested field support"""
        if not hasattr(item, "children") or not item.children:
            return str(item), None

        # According to grammar: projectionItem : expr ( AS? symbolPrimitive )? ;
        # children[0] is always the expression
        # If there's an alias, children[1] might be AS and children[2] symbolPrimitive
        # OR children[1] might be just symbolPrimitive (without AS)

        field_name = item.children[0].getText()
        # Normalize bracket notation (jmspath) to Mongo dot notation
        field_name = self.normalize_field_path(field_name)

        alias = None

        if len(item.children) >= 2:
            # Check if we have an alias
            if len(item.children) == 3:
                # Pattern: expr AS symbolPrimitive
                if hasattr(item.children[1], "getText") and item.children[1].getText().upper() == "AS":
                    alias = item.children[2].getText()
            elif len(item.children) == 2:
                # Pattern: expr symbolPrimitive (without AS)
                alias = item.children[1].getText()

        return field_name, self.unquote_identifier(alias)


class FromHandler(BaseHandler):
    """Handles FROM clause parsing with support for regular collections and aggregate() function calls"""

    def can_handle(self, ctx: Any) -> bool:
        """Check if this is a from context"""
        return hasattr(ctx, "tableReference")

    @staticmethod
    def _strip_collection_quotes(name: str) -> str:
        """Strip surrounding double quotes from collection name if present.

        Args:
            name: Collection name, potentially quoted

        Returns:
            Collection name with quotes removed
        """
        return re.sub(r'^"([^"]+)"$', r"\1", name)

    def _parse_function_call(self, ctx: Any) -> Optional[Dict[str, Any]]:
        """
        Detect and parse aggregate() function calls in FROM clause.

        Supports:
        - collection.aggregate('pipeline_json', 'options_json')
        - aggregate('pipeline_json', 'options_json')

        Returns dict with:
        - function_name: 'aggregate'
        - collection: collection name (or None if unqualified)
        - pipeline: JSON string for pipeline
        - options: JSON string for options
        """
        try:
            # Get the tableReference from FROM clause
            if not hasattr(ctx, "tableReference"):
                return None

            table_ref = ctx.tableReference()
            if not table_ref:
                return None

            # Get the text to analyze
            text = table_ref.getText() if hasattr(table_ref, "getText") else str(table_ref)

            # Pattern: [qualifier.]functionName(arg1, arg2)
            # We need to match: (optional_collection.)aggregate('...', '...')
            # Support collection names with double quotes for special characters like hyphens
            pattern = r"^(?:(\"[^\"]+\"|\w+)\.)?aggregate\s*\(\s*'([^']*)'\s*,\s*'([^']*)'\s*\)$"
            match = re.match(pattern, text, re.IGNORECASE | re.DOTALL)

            if not match:
                return None

            collection = match.group(1)  # Can be None for unqualified aggregate()
            # Strip quotes from collection name if present
            if collection:
                collection = self._strip_collection_quotes(collection)
            pipeline = match.group(2)
            options = match.group(3)

            _logger.debug(
                f"Detected aggregate call: collection={collection}, pipeline={pipeline[:50]}..., options={options}"
            )

            return {
                "function_name": "aggregate",
                "collection": collection,
                "pipeline": pipeline,
                "options": options,
            }
        except Exception as e:
            _logger.debug(f"Error parsing function call: {e}")
            return None

    @staticmethod
    def _collection_reference(table_ref: Any) -> Tuple[Optional[str], Optional[str], Optional[str]]:
        """Return (collection text, alias, problem) for a FROM table reference.

        Only a single collection, optionally aliased, can be translated. Joins, subqueries
        and AT/BY bindings are reported as a problem so the query fails instead of reading
        a collection named after the whole clause.
        """
        if not hasattr(table_ref, "getRuleIndex"):
            return table_ref.getText(), None, None  # not a parse-tree node: a plain name
        while isinstance(table_ref, PartiQLParser.TableWrappedContext):
            table_ref = table_ref.tableReference()
        if not isinstance(table_ref, PartiQLParser.TableRefBaseContext):
            return None, None, "FROM with a join"
        base = table_ref.tableNonJoin().tableBaseReference()
        if isinstance(base, PartiQLParser.TableBaseRefSymbolContext):
            source, alias = base.source, base.symbolPrimitive().getText()
        elif isinstance(base, PartiQLParser.TableBaseRefClausesContext):
            if base.atIdent() is not None or base.byIdent() is not None:
                return None, None, "FROM ... AT/BY"
            source = base.source
            alias = base.asIdent().symbolPrimitive().getText() if base.asIdent() is not None else None
        else:
            return None, None, "FROM with UNPIVOT or a graph match"
        text = source.getText()
        if text.startswith("("):
            return None, None, "FROM a subquery (use mode=superset)"
        return text, alias, None

    def handle_visitor(self, ctx: PartiQLParser.FromClauseContext, parse_result: "QueryParseResult") -> Any:
        """Handle FROM clause - detect aggregate calls or regular collections"""
        if hasattr(ctx, "tableReference") and ctx.tableReference():
            # Try to detect aggregate function call
            func_info = self._parse_function_call(ctx)

            if func_info and func_info["function_name"] == "aggregate":
                # Mark as aggregate query
                if hasattr(parse_result, "is_aggregate_query"):
                    parse_result.is_aggregate_query = True
                if hasattr(parse_result, "aggregate_pipeline"):
                    parse_result.aggregate_pipeline = func_info["pipeline"]
                if hasattr(parse_result, "aggregate_options"):
                    parse_result.aggregate_options = func_info["options"]

                # Set collection name if qualified, otherwise it's collection-agnostic
                if func_info["collection"]:
                    parse_result.collection = func_info["collection"]

                _logger.info(f"Parsed aggregate call: collection={func_info['collection']}")
                return func_info

            # Regular collection reference, optionally aliased
            source, alias, problem = self._collection_reference(ctx.tableReference())
            if problem:
                parse_result.unsupported_clauses.append(problem)
                return None
            # Strip surrounding quotes from collection name (e.g., "user.accounts" -> user.accounts)
            collection_name = self._strip_collection_quotes(source)
            parse_result.collection = collection_name
            parse_result.collection_alias = ContextUtilsMixin.unquote_identifier(alias) if alias else None
            _logger.debug(f"Parsed regular collection: {collection_name} (alias {alias})")
            return collection_name

        return None


class WhereHandler(BaseHandler):
    """Handles WHERE clause parsing"""

    def __init__(self):
        self._expression_handler = EnhancedWhereHandler()

    def can_handle(self, ctx: Any) -> bool:
        """Check if this is a where context"""
        return hasattr(ctx, "exprSelect")

    def handle_visitor(self, ctx: PartiQLParser.WhereClauseSelectContext, parse_result: "QueryParseResult") -> Any:
        if hasattr(ctx, "exprSelect") and ctx.exprSelect():
            from .where_tree import WhereTreeBuilder

            # Translate over the parse tree with SQL three-valued logic. A clause that
            # cannot be translated fails the query; it never falls back to a partial
            # or text-search filter that would return different rows.
            try:
                parse_result.filter_conditions = WhereTreeBuilder().build(ctx.exprSelect())
            except Exception as e:
                parse_result.unsupported_clauses.append(f"WHERE ({e})")
                parse_result.filter_conditions = {}
            return parse_result.filter_conditions
        return {}
