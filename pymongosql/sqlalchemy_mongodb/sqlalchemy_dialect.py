# -*- coding: utf-8 -*-
import logging
import uuid
from typing import Any, Dict, List, Optional, Tuple, Type
from urllib.parse import quote_plus

from sqlalchemy import pool, types
from sqlalchemy.engine import default, url
from sqlalchemy.sql import compiler, sqltypes
from sqlalchemy.sql.sqltypes import NULLTYPE

import pymongosql

_logger = logging.getLogger(__name__)

try:
    import sqlalchemy

    SQLALCHEMY_VERSION = tuple(map(int, sqlalchemy.__version__.split(".")[:2]))
    SQLALCHEMY_2X = SQLALCHEMY_VERSION >= (2, 0)
except ImportError:
    SQLALCHEMY_VERSION = (1, 4)  # Default fallback
    SQLALCHEMY_2X = False

# Version-specific imports
if SQLALCHEMY_2X:
    try:
        from sqlalchemy.engine.interfaces import Dialect
    except ImportError:
        # Fallback for different 2.x versions
        from sqlalchemy.engine.default import DefaultDialect as Dialect
else:
    from sqlalchemy.engine.interfaces import Dialect


def _partiql_keywords() -> set:
    """Lower-case PartiQL keywords, e.g. ``count`` or ``value``.

    The grammar rejects a keyword used as a bare identifier (``COUNT(*) AS count``
    is a syntax error), so SQLAlchemy must quote these names.
    """
    import re

    from pymongosql.sql.partiql.PartiQLLexer import PartiQLLexer

    return {name.strip("'").lower() for name in PartiQLLexer.literalNames if re.fullmatch(r"'[A-Za-z_]+'", name or "")}


class PyMongoSQLIdentifierPreparer(compiler.IdentifierPreparer):
    """MongoDB-specific identifier preparer.

    MongoDB collection and field names have specific rules that differ
    from SQL databases.
    """

    reserved_words = (
        set(
            [
                # MongoDB reserved words and operators
                "$eq",
                "$ne",
                "$gt",
                "$gte",
                "$lt",
                "$lte",
                "$in",
                "$nin",
                "$and",
                "$or",
                "$not",
                "$nor",
                "$exists",
                "$type",
                "$mod",
                "$regex",
                "$text",
                "$where",
                "$all",
                "$elemMatch",
                "$size",
                "$bitsAllClear",
                "$bitsAllSet",
                "$bitsAnyClear",
                "$bitsAnySet",
            ]
        )
        | _partiql_keywords()
    )

    def __init__(self, dialect: Dialect, **kwargs: Any) -> None:
        super().__init__(dialect, **kwargs)
        # MongoDB allows most characters in field names - use regex pattern
        import re

        self.legal_characters = re.compile(r"^[$a-zA-Z0-9_.]+$")


class PyMongoSQLCompiler(compiler.SQLCompiler):
    """MongoDB-specific SQL compiler.

    Handles SQL compilation specific to MongoDB's query patterns.
    """

    def visit_column(self, column, include_table=True, **kwargs):
        """Render column references without a table qualifier.

        A statement reads a single collection, and PyMongoSQL resolves a dotted
        reference such as ``users.name`` as the embedded-document path ``name``
        inside a field ``users``. SQLAlchemy qualifies every table-bound column,
        so a qualified reference would silently read NULL.
        """
        return super().visit_column(column, include_table=False, **kwargs)

    def visit_like_op_binary(self, binary, operator, **kw):
        """Render LIKE patterns inline: PyMongoSQL turns them into a regex while parsing."""
        kw["literal_binds"] = True
        return super().visit_like_op_binary(binary, operator, **kw)

    def visit_not_like_op_binary(self, binary, operator, **kw):
        kw["literal_binds"] = True
        return super().visit_not_like_op_binary(binary, operator, **kw)


class PyMongoSQLDDLCompiler(compiler.DDLCompiler):
    """MongoDB-specific DDL compiler.

    Handles Data Definition Language operations for MongoDB.
    """

    def visit_create_table(self, create, **kwargs):
        """Handle CREATE TABLE - MongoDB creates collections on first insert."""
        # MongoDB collections are created implicitly
        return "-- Collection will be created on first insert"

    def visit_drop_table(self, drop, **kwargs):
        """Handle DROP TABLE - translates to MongoDB collection drop."""
        table = drop.element
        return f"-- DROP COLLECTION {self.preparer.format_table(table)}"


class PyMongoSQLTypeCompiler(compiler.GenericTypeCompiler):
    """MongoDB-specific type compiler.

    Handles type mapping between SQL types and MongoDB BSON types.
    """

    def visit_VARCHAR(self, type_, **kwargs):
        return "STRING"

    def visit_CHAR(self, type_, **kwargs):
        return "STRING"

    def visit_TEXT(self, type_, **kwargs):
        return "STRING"

    def visit_INTEGER(self, type_, **kwargs):
        return "INT32"

    def visit_BIGINT(self, type_, **kwargs):
        return "INT64"

    def visit_FLOAT(self, type_, **kwargs):
        return "DOUBLE"

    def visit_NUMERIC(self, type_, **kwargs):
        return "DECIMAL128"

    def visit_DECIMAL(self, type_, **kwargs):
        return "DECIMAL128"

    def visit_DATETIME(self, type_, **kwargs):
        return "DATE"

    def visit_DATE(self, type_, **kwargs):
        return "DATE"

    def visit_BOOLEAN(self, type_, **kwargs):
        return "BOOL"


def _decode_decimal128(processor):
    """Wrap a Numeric result processor so it also accepts BSON Decimal128."""
    from bson import Decimal128

    def process(value):
        if isinstance(value, Decimal128):
            value = value.to_decimal()
        return processor(value) if processor else value

    return process


class _MongoNumeric(sqltypes.Numeric):
    """Numeric that returns ``decimal.Decimal`` (or float) for Decimal128 values."""

    def result_processor(self, dialect, coltype):
        return _decode_decimal128(super().result_processor(dialect, coltype))


class _MongoInteger(sqltypes.Integer):
    """Integer that returns ``int`` for BSON int64 values (``bson.Int64``)."""

    def result_processor(self, dialect, coltype):
        from bson import Int64

        def process(value):
            return int(value) if isinstance(value, Int64) else value

        return process


class _MongoFloat(sqltypes.Float):
    """Float that returns ``float`` (or Decimal) for Decimal128 values."""

    def result_processor(self, dialect, coltype):
        return _decode_decimal128(super().result_processor(dialect, coltype))


class _MongoUuid(getattr(sqltypes, "Uuid", sqltypes.TypeEngine)):  # Uuid is new in SQLAlchemy 2.0
    """Uuid stored as BSON binary subtype 4 (the standard UUID representation).

    PyMongo returns ``uuid.UUID`` under ``uuidRepresentation=standard`` and a
    subtype-4 ``Binary`` otherwise; the generic non-native Uuid processors expect a
    hex string and fail on both. Legacy subtype 3 is left as ``Binary``: its byte
    order depends on the driver that wrote it.
    """

    def bind_processor(self, dialect):
        from bson.binary import Binary

        def process(value):
            if value is None:
                return None
            if not isinstance(value, uuid.UUID):
                value = uuid.UUID(str(value))
            return Binary.from_uuid(value)

        return process

    def result_processor(self, dialect, coltype):
        from bson.binary import UUID_SUBTYPE, Binary

        def process(value):
            if isinstance(value, Binary) and value.subtype == UUID_SUBTYPE:
                value = value.as_uuid()
            elif isinstance(value, str):
                value = uuid.UUID(value)
            if isinstance(value, uuid.UUID) and not self.as_uuid:
                return str(value)
            return value

        return process


class PyMongoSQLDialect(default.DefaultDialect):
    """SQLAlchemy dialect for PyMongoSQL.

    This dialect enables PyMongoSQL to work with SQLAlchemy by providing
    the necessary interface methods and compilation logic.

    Compatible with SQLAlchemy 1.4+ and 2.x versions.
    """

    name = "mongodb"
    driver = "pymongosql"

    # Version compatibility
    _sqlalchemy_version = SQLALCHEMY_VERSION
    _is_sqlalchemy_2x = SQLALCHEMY_2X

    # DB API 2.0 compliance
    supports_alter = False  # MongoDB doesn't support ALTER TABLE
    supports_comments = False  # No SQL comments in MongoDB
    supports_default_values = True
    supports_empty_inserts = True
    supports_multivalues_insert = True
    supports_native_decimal = True  # BSON Decimal128
    # PyMongo returns Decimal128, not decimal.Decimal; convert on the way out.
    supports_native_uuid = True  # BSON binary subtype 4
    colspecs = {sqltypes.Numeric: _MongoNumeric, sqltypes.Float: _MongoFloat, sqltypes.Integer: _MongoInteger}
    if hasattr(sqltypes, "Uuid"):
        colspecs[sqltypes.Uuid] = _MongoUuid
    supports_native_boolean = True  # BSON Boolean
    supports_sequences = False  # No sequences in MongoDB
    supports_native_enum = False  # No native enums

    # MongoDB-specific features
    supports_statement_cache = True
    supports_server_side_cursors = True

    # Connection characteristics
    poolclass = pool.StaticPool

    # Compilation
    statement_compiler = PyMongoSQLCompiler
    ddl_compiler = PyMongoSQLDDLCompiler
    type_compiler = PyMongoSQLTypeCompiler
    preparer = PyMongoSQLIdentifierPreparer

    # Default parameter style
    paramstyle = "qmark"  # Matches PyMongoSQL's paramstyle

    @classmethod
    def dbapi(cls):
        """Return the PyMongoSQL DBAPI module (SQLAlchemy 1.x compatibility)."""
        return pymongosql

    @classmethod
    def import_dbapi(cls):
        """Return the PyMongoSQL DBAPI module (SQLAlchemy 2.x)."""
        return pymongosql

    def _get_dbapi_module(self):
        """Internal method to get DBAPI module for instance access."""
        return pymongosql

    def __getattribute__(self, name):
        """Override getattribute to handle DBAPI access properly."""
        if name == "dbapi":
            # Always return the module directly for DBAPI access
            return pymongosql
        return super().__getattribute__(name)

    @staticmethod
    def _normalize_collection_name(statement: str) -> str:
        """Extract a collection name from the compiler's DROP COLLECTION placeholder."""
        collection_name = statement[len("-- DROP COLLECTION ") :].strip()
        if collection_name.startswith('"') and collection_name.endswith('"'):
            return collection_name[1:-1]
        return collection_name

    def _handle_ddl_placeholder(self, cursor, statement: str) -> bool:
        """Handle SQLAlchemy DDL placeholders without routing them through the SQL parser."""
        if statement == "-- Collection will be created on first insert":
            return True

        if statement.startswith("-- DROP COLLECTION "):
            cursor.connection.database.drop_collection(self._normalize_collection_name(statement))
            return True

        return False

    def do_execute(self, cursor, statement, parameters, context=None):
        """Execute statements, handling MongoDB DDL placeholders directly."""
        if self._handle_ddl_placeholder(cursor, statement):
            return None
        return super().do_execute(cursor, statement, parameters, context=context)

    def do_execute_no_params(self, cursor, statement, context=None):
        """Execute parameterless statements, handling MongoDB DDL placeholders directly."""
        if self._handle_ddl_placeholder(cursor, statement):
            return None
        return super().do_execute_no_params(cursor, statement, context=context)

    def create_connect_args(self, url: url.URL) -> Tuple[List[Any], Dict[str, Any]]:
        """Create connection arguments from SQLAlchemy URL.

        Supports standard MongoDB connection strings (mongodb://).
        Note: For mongodb+srv URLs, use them directly as connection strings
        rather than through SQLAlchemy create_engine due to SQLAlchemy parsing limitations.

        Args:
            url: SQLAlchemy URL object with MongoDB connection string

        Returns:
            Tuple of (args, kwargs) for PyMongoSQL connection
        """
        opts = {}

        # For MongoDB URLs, reconstruct the full URI to pass to PyMongoSQL
        # This ensures proper MongoDB connection string format
        uri_parts = []

        # Start with scheme (mongodb only - srv handled separately)
        uri_parts.append(f"{url.drivername}://")

        # SQLAlchemy has already percent-decoded the credentials, so they must be
        # re-escaped (RFC 3986) before pymongo parses the rebuilt URI.
        if url.username:
            if url.password:
                uri_parts.append(f"{quote_plus(url.username)}:{quote_plus(url.password)}@")
            else:
                uri_parts.append(f"{quote_plus(url.username)}@")

        # Add host and port
        if url.host:
            uri_parts.append(url.host)
            if url.port:
                uri_parts.append(f":{url.port}")

        # Add database
        if url.database:
            uri_parts.append(f"/{url.database}")

        # Add query parameters
        if url.query:
            query_parts = []
            for key, value in url.query.items():
                values = value if isinstance(value, (list, tuple)) else (value,)
                for item in values:
                    query_parts.append(f"{quote_plus(key)}={quote_plus(str(item))}")
            if query_parts:
                uri_parts.append(f"?{'&'.join(query_parts)}")

        # Pass the full MongoDB URI to PyMongoSQL
        mongodb_uri = "".join(uri_parts)
        opts["host"] = mongodb_uri

        return [], opts

    def get_schema_names(self, connection, **kwargs):
        """Get list of databases (schemas in SQL terms)."""
        # Use MongoDB admin command directly instead of SQL SHOW DATABASES
        try:
            # Access the underlying MongoDB client through the connection
            db_connection = connection.connection
            if hasattr(db_connection, "_client"):
                admin_db = db_connection._client.admin
                result = admin_db.command("listDatabases")
                return [db["name"] for db in result.get("databases", [])]
        except Exception as e:
            _logger.warning(f"Failed to get database names: {e}")
        return ["default"]  # Fallback to default database

    def has_table(self, connection, table_name: str, schema: Optional[str] = None, **kwargs) -> bool:
        """Check if a collection (table) exists."""
        try:
            # Use MongoDB listCollections command directly
            db_connection = connection.connection
            if hasattr(db_connection, "_client"):
                if schema:
                    db = db_connection._client[schema]
                else:
                    db = db_connection.database

                # Use listCollections command
                collections = db.list_collection_names()
                return table_name in collections
        except Exception as e:
            _logger.warning(f"Failed to check table existence: {e}")
        return False

    def get_table_names(self, connection, schema: Optional[str] = None, **kwargs) -> List[str]:
        """Get list of collections (tables), excluding views."""
        try:
            db_connection = connection.connection
            if hasattr(db_connection, "_client"):
                if schema:
                    db = db_connection._client[schema]
                else:
                    db = db_connection.database

                return db.list_collection_names(filter={"type": {"$ne": "view"}})
        except Exception as e:
            _logger.warning(f"Failed to get table names: {e}")
        return []

    def get_view_names(self, connection, schema: Optional[str] = None, **kwargs) -> List[str]:
        """Get list of MongoDB views."""
        try:
            db_connection = connection.connection
            if hasattr(db_connection, "_client"):
                if schema:
                    db = db_connection._client[schema]
                else:
                    db = db_connection.database

                return db.list_collection_names(filter={"type": "view"})
        except Exception as e:
            _logger.warning(f"Failed to get view names: {e}")
        return []

    def get_columns(self, connection, table_name: str, schema: Optional[str] = None, **kwargs) -> List[Dict[str, Any]]:
        """Get column information for a collection.

        MongoDB is schemaless, so this inspects documents to infer structure.
        """
        columns = []
        try:
            # Use direct MongoDB operations to sample documents and infer schema
            db_connection = connection.connection
            if hasattr(db_connection, "_client"):
                if schema:
                    db = db_connection._client[schema]
                else:
                    db = db_connection.database

                collection = db[table_name]

                # Sample a few documents to infer schema
                sample_docs = list(collection.find().limit(10))
                if sample_docs:
                    # Collect all unique field names and types. A null only
                    # types a field that no sampled document gives a value.
                    field_types = {}
                    for doc in sample_docs:
                        for field_name, value in doc.items():
                            if field_types.get(field_name, "null") == "null":
                                field_types[field_name] = self._infer_bson_type(value)

                    # Convert to SQLAlchemy column format
                    for field_name, bson_type in field_types.items():
                        columns.append(
                            {
                                "name": field_name,
                                "type": self._get_column_type(bson_type),
                                "nullable": field_name != "_id",  # _id is always required
                                "default": None,
                            }
                        )
                else:
                    # Empty collection, provide minimal _id column
                    columns = [
                        {
                            "name": "_id",
                            "type": types.String(),
                            "nullable": False,
                            "default": None,
                        }
                    ]

        except Exception as e:
            _logger.warning(f"Failed to get column info for {table_name}: {e}")
            # Fallback: provide minimal _id column
            columns = [
                {
                    "name": "_id",
                    "type": types.String(),
                    "nullable": False,
                    "default": None,
                }
            ]

        return columns

    def _infer_bson_type(self, value: Any) -> str:
        """Infer BSON type from a Python value."""
        from datetime import datetime

        from bson import Binary, Decimal128, Int64, ObjectId

        if isinstance(value, ObjectId):
            return "objectId"
        elif isinstance(value, str):
            return "string"
        elif isinstance(value, bool):
            return "bool"
        elif isinstance(value, Int64):
            return "long"
        elif isinstance(value, int):
            return "int"
        elif isinstance(value, Decimal128):
            return "decimal"
        elif isinstance(value, (Binary, bytes)):
            return "binData"
        elif isinstance(value, float):
            return "double"
        elif isinstance(value, datetime):
            return "date"
        elif isinstance(value, list):
            return "array"
        elif isinstance(value, dict):
            return "object"
        elif value is None:
            return "null"
        else:
            return "string"  # Default fallback

    def _get_column_type(self, mongo_type: str) -> Type[types.TypeEngine]:
        """Map MongoDB/BSON types to SQLAlchemy types."""
        type_map = {
            "objectId": types.String,
            "string": types.String,
            "int": types.Integer,
            "long": types.BigInteger,
            "double": types.Float,
            "decimal": types.DECIMAL,
            "bool": types.Boolean,
            "date": types.DateTime,
            "null": NULLTYPE,
            "array": types.JSON,
            "object": types.JSON,
            "binData": types.LargeBinary,
        }
        return type_map.get(mongo_type, types.String)

    def get_pk_constraint(self, connection, table_name: str, schema: Optional[str] = None, **kwargs) -> Dict[str, Any]:
        """Get primary key constraint info.

        MongoDB always has _id as the primary key.
        """
        return {"constrained_columns": ["_id"], "name": "pk_id"}

    def get_foreign_keys(
        self, connection, table_name: str, schema: Optional[str] = None, **kwargs
    ) -> List[Dict[str, Any]]:
        """Get foreign key constraints.

        MongoDB doesn't enforce foreign keys, return empty list.
        """
        return []

    def get_indexes(self, connection, table_name: str, schema: Optional[str] = None, **kwargs) -> List[Dict[str, Any]]:
        """Get index information for a collection."""
        indexes = []
        try:
            # Use direct MongoDB operations to get indexes
            db_connection = connection.connection
            if hasattr(db_connection, "_client"):
                if schema:
                    db = db_connection._client[schema]
                else:
                    db = db_connection.database

                collection = db[table_name]

                # Get index information
                index_info = collection.index_information()
                for index_name, index_spec in index_info.items():
                    # Extract column names from key specification
                    column_names = [field[0] for field in index_spec.get("key", [])]

                    indexes.append(
                        {
                            "name": index_name,
                            "column_names": column_names,
                            "unique": index_spec.get("unique", False),
                        }
                    )
        except Exception as e:
            _logger.warning(f"Failed to get index info for {table_name}: {e}")
            # Always include the default _id index as fallback
            indexes = [
                {
                    "name": "_id_",
                    "column_names": ["_id"],
                    "unique": True,
                }
            ]

        return indexes

    def do_rollback(self, dbapi_connection):
        """Rollback transaction.

        MongoDB has limited transaction support.
        """
        # PyMongoSQL should handle this
        if hasattr(dbapi_connection, "rollback"):
            try:
                dbapi_connection.rollback()
            except Exception:
                # MongoDB doesn't always support rollback - ignore errors
                # This is normal behavior for MongoDB connections without active transactions
                pass

    def do_commit(self, dbapi_connection):
        """Commit transaction.

        MongoDB auto-commits most operations.
        """
        # PyMongoSQL should handle this
        if hasattr(dbapi_connection, "commit"):
            try:
                dbapi_connection.commit()
            except Exception:
                # MongoDB auto-commits most operations - ignore errors
                # This is normal behavior for MongoDB connections
                pass

    def do_ping(self, dbapi_connection):
        """Ping the database to test connection status.

        Used by SQLAlchemy and tools like Superset for connection testing.
        This avoids the need to execute "SELECT 1" queries that would fail
        due to PartiQL grammar requirements.
        """
        if hasattr(dbapi_connection, "test_connection") and callable(dbapi_connection.test_connection):
            return dbapi_connection.test_connection()
        else:
            # Fallback: try to execute a simple ping command directly
            try:
                if hasattr(dbapi_connection, "_client"):
                    dbapi_connection._client.admin.command("ping")
                    return True
            except Exception:
                pass
            return False


# Version information
__sqlalchemy_version__ = SQLALCHEMY_VERSION
__supports_sqlalchemy_2x__ = SQLALCHEMY_2X
