# -*- coding: utf-8 -*-
import datetime
import logging
import sqlite3
from decimal import Decimal
from typing import Any, Dict, List, Optional

from ..error import NotSupportedError
from . import exact_decimal
from .query_db import QueryDatabase
from .time_grain import date_trunc_text, datetime_positions, datetime_text, parse_text, str_to_datetime_text

_logger = logging.getLogger(__name__)

# SQLite has no boolean type. Boolean columns are declared with this private type
# (NUMERIC affinity, stored as 0/1) and converted back to bool when a query selects
# the column itself; expressions over it (SUM, CASE, ...) keep their numeric result.
BOOLEAN_DECLTYPE = "PYMONGOSQL_BOOL"
sqlite3.register_converter(BOOLEAN_DECLTYPE, lambda raw: int(raw) != 0)
# Datetimes are stored as fixed-width UTC text (so text order is time order) and read
# back as datetime when a query selects the column itself.
DATETIME_DECLTYPE = "PYMONGOSQL_DATETIME"
sqlite3.register_converter(DATETIME_DECLTYPE, lambda raw: datetime.datetime.fromisoformat(raw.decode()))
_NUMERIC = ("INTEGER", "REAL")


class SQLiteTypeMapper:
    """Maps Python/MongoDB data types to SQLite3 types"""

    # Type mapping from Python types to SQLite3 types
    TYPE_MAP = {
        str: "TEXT",
        int: "INTEGER",
        float: "REAL",
        bool: BOOLEAN_DECLTYPE,  # stored as 0/1, read back as bool
        Decimal: "REAL",  # approximate copy; the exact text is kept in a side table
        datetime.datetime: DATETIME_DECLTYPE,
        bytes: "BLOB",
        type(None): "NULL",
        dict: "TEXT",  # Store as JSON string
        list: "TEXT",  # Store as JSON string
    }

    @classmethod
    def get_sqlite_type(cls, value: Any) -> str:
        """Get SQLite type for a Python value"""
        if value is None:
            return "NULL"

        value_type = type(value)
        if value_type in cls.TYPE_MAP:
            return cls.TYPE_MAP[value_type]

        # Default to TEXT for unknown types
        return "TEXT"

    @classmethod
    def infer_schema(cls, records: List[Dict[str, Any]]) -> Dict[str, str]:
        """
        Infer SQLite schema from a list of records.

        Args:
            records: List of dictionaries with data

        Returns:
            Dictionary mapping column names to SQLite types
        """
        schema = {}

        for record in records:
            for col_name, value in record.items():
                new_type = cls.get_sqlite_type(value)
                current = schema.get(col_name, "NULL")
                if current == "NULL":
                    # First non-null value determines the type; NULL fits every type
                    schema[col_name] = new_type
                elif new_type in _NUMERIC and current in _NUMERIC:
                    schema[col_name] = "REAL" if "REAL" in (new_type, current) else "INTEGER"
                elif new_type not in ("NULL", current):
                    # Upgrade to TEXT if types differ (safest option)
                    schema[col_name] = "TEXT"

        return schema

    @classmethod
    def convert_value(cls, value: Any, target_type: str) -> Any:
        """Convert value to appropriate SQLite type"""
        if value is None:
            return None

        if target_type == DATETIME_DECLTYPE:
            return datetime_text(value)
        if target_type in ("INTEGER", BOOLEAN_DECLTYPE):
            return int(value) if value is not None else None
        elif target_type == "REAL":
            return float(value) if value is not None else None
        elif target_type == "TEXT":
            if isinstance(value, (dict, list)):
                import json

                return json.dumps(value)
            return str(value)
        elif target_type == "BLOB":
            if isinstance(value, bytes):
                return value
            return str(value).encode()

        return value


class QueryDBSQLite(QueryDatabase):
    """Manages SQLite3 in-memory database for query database operations.

    This is the default implementation of QueryDatabase using SQLite3.
    Other RDBMS backends can be created by implementing the QueryDatabase interface.
    """

    def __init__(self) -> None:
        """Initialize SQLite3 bridge with in-memory database"""
        self._connection: Optional[sqlite3.Connection] = None
        self._tables: Dict[str, Dict[str, str]] = {}  # table_name -> schema
        self._exact_columns: Dict[str, List[str]] = {}  # table_name -> columns with decimals
        self._is_closed = False

    def _ensure_connection(self) -> sqlite3.Connection:
        """Ensure SQLite3 connection is available"""
        if self._is_closed:
            raise RuntimeError("SQLiteBridge is closed")

        if self._connection is None:
            # Create in-memory database
            self._connection = sqlite3.connect(":memory:", detect_types=sqlite3.PARSE_DECLTYPES)
            # Time-grain functions the engine spec's expressions use
            self._connection.create_function("date_trunc", 2, date_trunc_text, deterministic=True)
            self._connection.create_function("str_to_datetime", -1, str_to_datetime_text, deterministic=True)
            exact_decimal.register(self._connection)
            # Enable row factory to get dict-like rows
            self._connection.row_factory = sqlite3.Row
            _logger.debug("Created in-memory SQLite3 database")

        return self._connection

    def create_table(self, table_name: str, schema: Dict[str, str]) -> None:
        """
        Create a table in SQLite3.

        Args:
            table_name: Name of the table
            schema: Dictionary mapping column names to SQLite types
        """
        conn = self._ensure_connection()

        # Build CREATE TABLE statement
        columns = ", ".join([f'"{col}" {dtype}' for col, dtype in schema.items()])
        create_sql = f"CREATE TABLE {table_name} ({columns})"

        try:
            conn.execute(create_sql)
            conn.commit()
            self._tables[table_name] = schema
            _logger.debug(f"Created SQLite3 table: {table_name}")
        except sqlite3.Error as e:
            _logger.error(f"Error creating table {table_name}: {e}")
            raise

    def insert_records(
        self, table_name: str, records: List[Dict[str, Any]], schema: Optional[Dict[str, str]] = None
    ) -> int:
        """
        Insert records into a SQLite3 table.

        Args:
            table_name: Name of the table
            records: List of dictionaries to insert
            schema: Optional schema (will be inferred if not provided)

        Returns:
            Number of records inserted
        """
        if not records:
            return 0

        conn = self._ensure_connection()

        # Create table if not exists
        if table_name not in self._tables:
            if schema is None:
                schema = SQLiteTypeMapper.infer_schema(records)
            self.create_table(table_name, schema)

        # Build INSERT statement
        columns = list(records[0].keys())
        placeholders = ", ".join(["?" for _ in columns])
        quoted = ", ".join('"%s"' % col.replace('"', '""') for col in columns)
        insert_sql = f'INSERT INTO "{table_name}" ({quoted}) VALUES ({placeholders})'

        # Convert values to appropriate types
        schema = self._tables[table_name]
        converted_records = []

        for record in records:
            converted_row = tuple(
                SQLiteTypeMapper.convert_value(record.get(col), schema.get(col, "TEXT")) for col in columns
            )
            converted_records.append(converted_row)

        try:
            first_rowid = conn.execute(f'SELECT COALESCE(MAX(rowid), 0) + 1 FROM "{table_name}"').fetchone()[0]
            conn.executemany(insert_sql, converted_records)
            self._write_exact_copies(table_name, columns, records, first_rowid)
            conn.commit()
            _logger.debug(f"Inserted {len(records)} records into {table_name}")
            return len(records)
        except sqlite3.Error as e:
            _logger.error(f"Error inserting records into {table_name}: {e}")
            raise

    def _write_exact_copies(
        self, table_name: str, columns: List[str], records: List[Dict[str, Any]], first_rowid: int
    ) -> None:
        """Keep the exact text of every column holding decimals (see exact_decimal)."""
        exact = []
        for col in columns:
            values = [r.get(col) for r in records if r.get(col) is not None]
            if any(isinstance(v, Decimal) for v in values) and all(
                isinstance(v, (int, float, Decimal)) and not isinstance(v, bool) for v in values
            ):
                exact.append(col)
        if not exact:
            return
        if table_name in self._exact_columns:
            raise NotSupportedError("Decimal columns can be loaded into a query table only once")
        self._exact_columns[table_name] = exact
        conn = self._ensure_connection()
        side = f'"{table_name}_exact"'
        conn.execute(f"CREATE TABLE {side} (rid INTEGER PRIMARY KEY, %s)" % ", ".join('"%s" TEXT' % c for c in exact))
        conn.executemany(
            f"INSERT INTO {side} VALUES ({', '.join('?' * (len(exact) + 1))})",
            [
                (rid,) + tuple(None if r.get(c) is None else str(r.get(c)) for c in exact)
                for rid, r in enumerate(records, start=first_rowid)
            ],
        )

    def execute_query(self, query: str) -> List[Dict[str, Any]]:
        """
        Execute a query against the SQLite3 database.

        Args:
            query: SQL query string

        Returns:
            List of dictionaries with query results
        """
        conn = self._ensure_connection()

        positions: List[int] = []
        dates = datetime_positions(query)
        for table, exact in self._exact_columns.items():
            if table in query:
                query, positions = exact_decimal.rewrite(query, table, exact, list(self._tables[table]))
        try:
            cursor = conn.execute(query)
            # Fetch all rows and convert from sqlite3.Row to dict
            rows = cursor.fetchall()

            column_names = [desc[0] for desc in cursor.description] if cursor.description else []

            def convert(index: int, value: Any) -> Any:
                if value is None:
                    return None
                if index in positions:
                    return Decimal(value)
                return parse_text(value) if index in dates else value

            return [
                {name: convert(index, value) for index, (name, value) in enumerate(zip(column_names, row))}
                for row in rows
            ]
        except sqlite3.Error as e:
            _logger.error(f"Error executing query: {e}")
            raise

    def execute_query_cursor(self, query: str) -> sqlite3.Cursor:
        """
        Execute a query and return cursor for manual iteration.

        Args:
            query: SQL query string

        Returns:
            SQLite3 cursor for iteration
        """
        conn = self._ensure_connection()
        return conn.execute(query)

    def table_exists(self, table_name: str) -> bool:
        """Check if a table exists in the database"""
        return table_name in self._tables

    def get_table_schema(self, table_name: str) -> Optional[Dict[str, str]]:
        """Get the schema of a table"""
        return self._tables.get(table_name)

    def list_tables(self) -> List[str]:
        """List all tables in the database"""
        return list(self._tables.keys())

    def drop_table(self, table_name: str) -> None:
        """Drop a table from the database"""
        if table_name not in self._tables:
            return

        conn = self._ensure_connection()
        try:
            conn.execute(f"DROP TABLE {table_name}")
            conn.commit()
            del self._tables[table_name]
            _logger.debug(f"Dropped table: {table_name}")
        except sqlite3.Error as e:
            _logger.error(f"Error dropping table {table_name}: {e}")
            raise

    def close(self) -> None:
        """Close the SQLite3 connection"""
        if self._connection is not None:
            try:
                self._connection.close()
                _logger.debug("Closed SQLite3 database connection")
            except sqlite3.Error as e:
                _logger.error(f"Error closing SQLite3 connection: {e}")
            finally:
                self._connection = None
                self._is_closed = True

    def __enter__(self) -> "QueryDBSQLite":
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit"""
        self.close()

    def __repr__(self) -> str:
        return f"QueryDBSQLite(tables={list(self._tables.keys())}, closed={self._is_closed})"
