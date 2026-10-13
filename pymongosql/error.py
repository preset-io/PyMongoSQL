# -*- coding: utf-8 -*-
__all__ = [
    "Error",
    "Warning",
    "SqlSyntaxError",
    "InterfaceError",
    "DatabaseError",
    "InternalError",
    "OperationalError",
    "ProgrammingError",
    "DataError",
    "NotSupportedError",
]


class Error(Exception):
    pass


class Warning(Exception):
    pass


class SqlSyntaxError(Error):
    pass


class InterfaceError(Error):
    pass


class DatabaseError(Error):
    pass


class InternalError(DatabaseError):
    pass


class OperationalError(DatabaseError):
    pass


class ProgrammingError(DatabaseError):
    pass


class _LikeProgrammingError(ProgrammingError):
    """LIKE caller errors that must survive executor exception wrapping."""


class IntegrityError(DatabaseError):
    pass


class DataError(DatabaseError):
    pass


class NotSupportedError(DatabaseError):
    pass
