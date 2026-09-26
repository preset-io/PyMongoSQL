# -*- coding: utf-8 -*-
import json
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

from bson import json_util

if TYPE_CHECKING:
    from .delete_builder import DeleteExecutionPlan
    from .delete_handler import DeleteParseResult
    from .insert_builder import InsertExecutionPlan
    from .insert_handler import InsertParseResult
    from .query_builder import QueryExecutionPlan
    from .query_handler import QueryParseResult
    from .update_builder import UpdateExecutionPlan
    from .update_handler import UpdateParseResult

_logger = logging.getLogger(__name__)


@dataclass
class ExecutionPlan:
    """Base class for execution plans (query, insert, etc.).

    Provides common attributes and shared validation helpers.
    """

    collection: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert plan to a serializable dictionary. Must be implemented by subclasses."""
        raise NotImplementedError()

    def validate_base(self) -> list[str]:
        """Common validation checks for all plans.

        Returns a list of error messages for the caller to aggregate and log.
        """
        errors: list[str] = []
        if not self.collection:
            errors.append("Collection name is required")
        return errors


class BuilderFactory:
    """Factory for creating builders for different operations."""

    @staticmethod
    def create_query_builder():
        """Create a builder for SELECT queries"""
        # Local import to avoid circular dependency during module import
        from .query_builder import MongoQueryBuilder

        return MongoQueryBuilder()

    @staticmethod
    def create_insert_builder():
        """Create a builder for INSERT queries"""
        # Local import to avoid circular dependency during module import
        from .insert_builder import MongoInsertBuilder

        return MongoInsertBuilder()

    @staticmethod
    def create_delete_builder():
        """Create a builder for DELETE queries"""
        # Local import to avoid circular dependency during module import
        from .delete_builder import MongoDeleteBuilder

        return MongoDeleteBuilder()

    @staticmethod
    def create_update_builder():
        """Create a builder for UPDATE queries"""
        # Local import to avoid circular dependency during module import
        from .update_builder import MongoUpdateBuilder

        return MongoUpdateBuilder()


class ExecutionPlanBuilder:
    """Builder class to create execution plans from parse results.

    This class decouples the AST visitor from execution plan creation,
    providing a clean separation between parsing and plan generation.
    """

    @staticmethod
    def build_from_parse_result(
        parse_result: Union["QueryParseResult", "InsertParseResult", "DeleteParseResult", "UpdateParseResult"],
        operation: str,
    ) -> Union["QueryExecutionPlan", "InsertExecutionPlan", "DeleteExecutionPlan", "UpdateExecutionPlan"]:
        """Build an execution plan from a parse result based on the operation type.

        Args:
            parse_result: The parse result from the AST visitor
            operation: The operation type ('select', 'insert', 'delete', or 'update')

        Returns:
            The appropriate execution plan for the operation

        Raises:
            SqlSyntaxError: If the parse result is invalid or plan generation fails
        """
        if operation == "insert":
            return ExecutionPlanBuilder._build_insert_plan(parse_result)
        elif operation == "delete":
            return ExecutionPlanBuilder._build_delete_plan(parse_result)
        elif operation == "update":
            return ExecutionPlanBuilder._build_update_plan(parse_result)
        else:  # Default to SELECT/query
            return ExecutionPlanBuilder._build_query_plan(parse_result)

    @staticmethod
    def _strip_collection_qualifier(parse_result: "QueryParseResult") -> None:
        """Resolve ``collection.field`` references to ``field``.

        SQL qualifies a column with the table it belongs to, while MongoDB reads a
        dotted name as an embedded-document path. Without this, a qualified
        reference such as ``users.name`` reads the missing path ``users.name``
        and silently returns NULL. As in SQL, the collection name takes
        precedence over an embedded document of the same name.
        """
        collection = parse_result.collection
        if not collection:
            return
        # With FROM users AS u, u.name is the column name; so is users.name
        prefixes = [f"{q}." for q in (parse_result.collection_alias, collection) if q]

        def strip(name: Any) -> Any:
            for prefix in prefixes:
                if isinstance(name, str) and name.startswith(prefix) and len(name) > len(prefix):
                    return name[len(prefix) :]
            return name

        def strip_filter(value: Any) -> Any:
            if isinstance(value, dict):
                return {strip(k): strip_filter(v) for k, v in value.items()}
            if isinstance(value, list):
                return [strip_filter(v) for v in value]
            return value

        parse_result.projection = {strip(k): v for k, v in parse_result.projection.items()}
        parse_result.column_aliases = {strip(k): v for k, v in parse_result.column_aliases.items()}
        parse_result.sort_fields = [{strip(k): v for k, v in spec.items()} for spec in parse_result.sort_fields]
        parse_result.filter_conditions = strip_filter(parse_result.filter_conditions)
        for func_info in parse_result.aggregate_functions:
            func_info["argument"] = strip(func_info["argument"])
        parse_result.group_by = [strip(name) for name in parse_result.group_by]
        for expression in parse_result.computed.values():
            expression["field"] = strip(expression["field"])
        for item in parse_result.select_items:
            if "field" in item:
                item["field"] = strip(item["field"])

    @staticmethod
    def _build_query_plan(parse_result: "QueryParseResult") -> "QueryExecutionPlan":
        """Build a query execution plan from SELECT parsing."""
        from ..error import NotSupportedError

        ExecutionPlanBuilder._strip_collection_qualifier(parse_result)
        if parse_result.unsupported_clauses:
            raise NotSupportedError(f"Unsupported SQL clause: {', '.join(parse_result.unsupported_clauses)}")

        # Auto-generate aggregate pipeline for SQL aggregate functions (COUNT, SUM, etc.), GROUP BY and HAVING
        if parse_result.aggregate_functions or parse_result.group_by or parse_result.having is not None:
            return ExecutionPlanBuilder._build_sql_aggregate_plan(parse_result)
        if any("computed" in item for item in parse_result.select_items):
            return ExecutionPlanBuilder._build_computed_plan(parse_result)

        # ORDER BY may name a column by its SELECT alias; find() sorts on the field
        field_for_alias = {alias: name for name, alias in parse_result.column_aliases.items()}
        sort_fields = [
            {field_for_alias.get(name, name): direction for name, direction in spec.items()}
            for spec in parse_result.sort_fields
        ]

        builder = BuilderFactory.create_query_builder().collection(parse_result.collection)

        builder.filter(parse_result.filter_conditions).project(parse_result.projection).column_aliases(
            parse_result.column_aliases
        ).sort(sort_fields).limit(parse_result.limit_value).skip(parse_result.offset_value)

        # Set aggregate flags BEFORE building (needed for validation)
        if hasattr(parse_result, "is_aggregate_query") and parse_result.is_aggregate_query:
            builder._execution_plan.is_aggregate_query = True
            builder._execution_plan.aggregate_pipeline = parse_result.aggregate_pipeline
            builder._execution_plan.aggregate_options = parse_result.aggregate_options

        # Now build and validate
        plan = builder.build()
        return plan

    @staticmethod
    def _build_computed_plan(parse_result: "QueryParseResult") -> "QueryExecutionPlan":
        """A SELECT with computed columns (DATE_TRUNC) and no grouping.

        Pipeline: $match (WHERE), $addFields (computed columns), $sort, then $project of
        the SELECT list in order; OFFSET/LIMIT are applied after binding.
        """
        from ..superset_mongodb.time_grain import mongo_expression

        builder = BuilderFactory.create_query_builder().collection(parse_result.collection)
        pipeline: List[Dict[str, Any]] = []
        if parse_result.filter_conditions:
            pipeline.append({"$match": parse_result.filter_conditions})
        added: Dict[str, Any] = {}
        project: Dict[str, Any] = {}
        sources: Dict[str, str] = {}  # names ORDER BY may use -> sortable field
        outputs = []
        for index, item in enumerate(parse_result.select_items):
            if "computed" in item:
                expression = parse_result.computed[item["computed"]]
                hidden = f"__computed{index}"
                added[hidden] = mongo_expression(expression["unit"], expression["field"])
                source, text = hidden, item["computed"]
            else:
                source = text = item["field"]
            output = item["alias"] or text
            project[output] = f"${source}"
            sources[output] = sources[text] = sources[text.upper()] = source
            outputs.append(output)
        if "_id" not in project:
            project["_id"] = 0
        if added:
            pipeline.append({"$addFields": added})
        sort = {}
        for spec in parse_result.sort_fields:
            for name, direction in spec.items():
                sort[sources.get(name, sources.get(name.upper(), name))] = direction
        if sort:
            pipeline.append({"$sort": sort})
        pipeline.append({"$project": project})
        builder.skip(parse_result.offset_value).limit(parse_result.limit_value)
        builder._execution_plan.is_aggregate_query = True
        builder._execution_plan.aggregate_parameterized = True
        builder._execution_plan.aggregate_pipeline = json_util.dumps(pipeline)
        builder._execution_plan.aggregate_options = json.dumps({})
        builder._execution_plan.projection_stage = {name: 1 for name in outputs}
        return builder.build()

    @staticmethod
    def _aggregate_source(func_info: Dict[str, Any], key: str) -> Any:
        """$project expression for an accumulator: the value, or the reduced DISTINCT set."""
        if not func_info.get("distinct"):
            return f"${key}"
        # SQL ignores NULL in DISTINCT aggregates
        values = {"$setDifference": [f"${key}", [None]]}
        reducer = {"COUNT": "$size", "SUM": "$sum", "AVG": "$avg", "MIN": "$min", "MAX": "$max"}
        return {reducer[func_info["function"]]: values}

    @staticmethod
    def _translate_having(parse_result: "QueryParseResult", group_keys: Dict[str, str], hidden: Dict[str, Any]) -> Any:
        """Translate HAVING into a $match on the grouped outputs (SQL three-valued logic)."""
        from .partiql.PartiQLParser import PartiQLParser
        from .query_handler import SelectHandler
        from .where_tree import _PATH_NODES, WhereTreeBuilder, _field_path

        prefixes = [f"{q}." for q in (parse_result.collection_alias, parse_result.collection) if q]

        def strip(name: str) -> str:
            for prefix in prefixes:
                if name.startswith(prefix) and len(name) > len(prefix):
                    return name[len(prefix) :]
            return name

        outputs = {}
        for item in parse_result.select_items:
            if "aggregate" in item:
                info = parse_result.aggregate_functions[item["aggregate"]]
                outputs[(info["function"], info["argument"], bool(info.get("distinct")))] = info["alias"]
            else:
                outputs[item["field"]] = item["alias"] or item["field"]
        aliases = set(outputs.values())

        def resolve(node: Any) -> Any:
            if isinstance(node, (PartiQLParser.CountAllContext, PartiQLParser.AggregateBaseContext)):
                kind, detail = SelectHandler._classify_item(node)
                if kind != "aggregate":
                    raise ValueError(f"Unsupported aggregate in HAVING: {node.getText()}")
                func, arg, distinct = detail
                signature = (func, strip(arg) if arg != "*" else arg, distinct)
                if signature in outputs:
                    return outputs[signature]
                name = f"__having{len(hidden)}"
                parse_result.aggregate_functions.append(
                    {"function": func, "argument": signature[1], "distinct": distinct, "alias": name, "expression": ""}
                )
                hidden[name] = len(parse_result.aggregate_functions) - 1
                outputs[signature] = name
                return name
            if isinstance(node, _PATH_NODES):
                name = strip(_field_path(node))
                if name in aliases:
                    return name
                if name in outputs:
                    return outputs[name]
                if name in group_keys:
                    hidden_name = f"__having{len(hidden)}"
                    hidden[hidden_name] = f"$_id.{group_keys[name]}"
                    outputs[name] = hidden_name
                    return hidden_name
                raise ValueError(f"HAVING column '{name}' must be grouped, aggregated or a SELECT alias")
            return None

        return WhereTreeBuilder(resolver=resolve).build(parse_result.having)

    @staticmethod
    def _build_sql_aggregate_plan(parse_result: "QueryParseResult") -> "QueryExecutionPlan":
        """Build an aggregate execution plan from SQL aggregate functions and GROUP BY.

        Pipeline: $match (WHERE), $group (GROUP BY keys as _id, one accumulator per
        aggregate), $project (SELECT list, in order, under its output names), then
        $sort, $skip and $limit on those output names.
        """
        from ..error import NotSupportedError

        _FUNCTION_TO_ACCUMULATOR = {
            "COUNT": "$sum",
            "SUM": "$sum",
            "AVG": "$avg",
            "MIN": "$min",
            "MAX": "$max",
        }

        builder = BuilderFactory.create_query_builder().collection(parse_result.collection)

        pipeline = []

        # Add $match stage if there are filter conditions (from WHERE clause)
        if parse_result.filter_conditions:
            pipeline.append({"$match": parse_result.filter_conditions})

        from ..superset_mongodb.time_grain import mongo_expression

        group_keys = {name: f"g{i}" for i, name in enumerate(parse_result.group_by)}

        def key_source(name: str) -> Any:
            computed = parse_result.computed.get(name)
            return mongo_expression(computed["unit"], computed["field"]) if computed else f"${name}"

        group_stage = {"_id": {key: key_source(name) for name, key in group_keys.items()} if group_keys else None}

        # HAVING may name select-list outputs, grouped columns or aggregates; the ones
        # not in the SELECT list are computed as hidden outputs and removed afterwards.
        hidden: Dict[str, Any] = {}
        having_filter = None
        if parse_result.having is not None:
            having_filter = ExecutionPlanBuilder._translate_having(parse_result, group_keys, hidden)

        accumulator_keys = []
        for i, func_info in enumerate(parse_result.aggregate_functions):
            func_name = func_info["function"]
            arg = func_info["argument"]
            accumulator = _FUNCTION_TO_ACCUMULATOR[func_name]
            # $group output names may not contain "." or start with "$", nor repeat
            key = func_info["alias"]
            if func_info.get("distinct") or "." in key or key.startswith("$") or key == "_id" or key in group_stage:
                key = f"__agg{i}"
            accumulator_keys.append(key)

            if func_info.get("distinct"):
                # Collect the distinct values; the $project stage reduces the set
                group_stage[key] = {"$addToSet": f"${arg}"}
            elif func_name == "COUNT" and arg == "*":
                group_stage[key] = {accumulator: 1}
            elif func_name == "COUNT":
                # COUNT(field) counts documents where the field is present and not null
                group_stage[key] = {"$sum": {"$cond": [{"$gt": [f"${arg}", None]}, 1, 0]}}
            else:
                group_stage[key] = {accumulator: f"${arg}"}

        pipeline.append({"$group": group_stage})

        # Map every SELECT item, in order, to its output name and source
        project_stage = {"_id": 0}
        outputs = []
        output_for = {}  # names ORDER BY may use -> output name
        for item in parse_result.select_items:
            if "aggregate" in item:
                func_info = parse_result.aggregate_functions[item["aggregate"]]
                output, key = func_info["alias"], accumulator_keys[item["aggregate"]]
                source = 1 if key == output else ExecutionPlanBuilder._aggregate_source(func_info, key)
                output_for[func_info["expression"].upper()] = output
            else:
                name = item.get("field") or item["computed"]
                if name not in group_keys:
                    raise NotSupportedError(f"Column '{name}' must appear in GROUP BY or in an aggregate function")
                output, source = item["alias"] or name, f"$_id.{group_keys[name]}"
                output_for[name] = output
            output_for[output] = output
            project_stage[output] = source
            outputs.append(output)
        for name, source in hidden.items():
            if isinstance(source, int):  # a hidden aggregate: index into aggregate_functions
                project_stage[name] = ExecutionPlanBuilder._aggregate_source(
                    parse_result.aggregate_functions[source], accumulator_keys[source]
                )
            else:
                project_stage[name] = source
        pipeline.append({"$project": project_stage})
        if having_filter is not None:
            pipeline.append({"$match": having_filter})
            if hidden:
                pipeline.append({"$project": {name: 0 for name in hidden}})

        sort_stage = {}
        for spec in parse_result.sort_fields:
            for name, direction in spec.items():
                output = output_for.get(name, output_for.get(name.upper()))
                if output is None:
                    raise NotSupportedError(f"ORDER BY '{name}' must name a selected column or its alias")
                sort_stage[output] = direction
        if sort_stage:
            pipeline.append({"$sort": sort_stage})
        # OFFSET/LIMIT (integers or parameters) are applied by the executor after binding
        builder.skip(parse_result.offset_value).limit(parse_result.limit_value)

        # Configure the execution plan as an aggregate query
        builder._execution_plan.is_aggregate_query = True
        builder._execution_plan.aggregate_parameterized = True
        # Extended JSON keeps Decimal128/datetime literals and parameter markers intact
        builder._execution_plan.aggregate_pipeline = json_util.dumps(pipeline)
        builder._execution_plan.aggregate_options = json.dumps({})

        # Set projection for ResultSet description, in SELECT order
        builder._execution_plan.projection_stage = {name: 1 for name in outputs}

        plan = builder.build()
        return plan

    @staticmethod
    def _build_insert_plan(parse_result: "InsertParseResult") -> "InsertExecutionPlan":
        """Build an INSERT execution plan from INSERT parsing."""
        from ..error import SqlSyntaxError

        if parse_result.has_errors:
            raise SqlSyntaxError(parse_result.error_message or "INSERT parsing failed")

        builder = BuilderFactory.create_insert_builder().collection(parse_result.collection)

        documents = parse_result.insert_documents or []
        builder.insert_documents(documents)

        if parse_result.parameter_style:
            builder.parameter_style(parse_result.parameter_style)

        if parse_result.parameter_count > 0:
            builder.parameter_count(parse_result.parameter_count)

        return builder.build()

    @staticmethod
    def _build_delete_plan(parse_result: "DeleteParseResult") -> "DeleteExecutionPlan":
        """Build a DELETE execution plan from DELETE parsing."""
        from ..error import SqlSyntaxError

        if parse_result.has_errors:
            # An untranslated WHERE must never become an empty filter (every document)
            raise SqlSyntaxError(parse_result.error_message or "DELETE parsing failed")
        _logger.debug(
            f"Building DELETE plan with collection: {parse_result.collection}, "
            f"filters: {parse_result.filter_conditions}"
        )
        builder = BuilderFactory.create_delete_builder().collection(parse_result.collection)

        if parse_result.filter_conditions:
            builder.filter_conditions(parse_result.filter_conditions)

        return builder.build()

    @staticmethod
    def _build_update_plan(parse_result: "UpdateParseResult") -> "UpdateExecutionPlan":
        """Build an UPDATE execution plan from UPDATE parsing."""
        from ..error import SqlSyntaxError

        if parse_result.has_errors:
            # An untranslated WHERE must never become an empty filter (every document)
            raise SqlSyntaxError(parse_result.error_message or "UPDATE parsing failed")
        _logger.debug(
            f"Building UPDATE plan with collection: {parse_result.collection}, "
            f"update_fields: {parse_result.update_fields}, "
            f"filters: {parse_result.filter_conditions}"
        )
        builder = BuilderFactory.create_update_builder().collection(parse_result.collection)

        if parse_result.update_fields:
            builder.update_fields(parse_result.update_fields)

        if parse_result.filter_conditions:
            builder.filter_conditions(parse_result.filter_conditions)

        return builder.build()


__all__ = [
    "ExecutionPlan",
    "BuilderFactory",
    "ExecutionPlanBuilder",
]
