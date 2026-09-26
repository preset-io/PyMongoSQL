# -*- coding: utf-8 -*-
"""DATE_TRUNC(unit, value): the time-grain function Superset's engine spec emits.

The same function is evaluated in two places with the same meaning:

- on a collection, translated to MongoDB's ``$dateTrunc`` (see ``mongo_expression``);
- in the superset-mode SQLite stage, registered as a SQL function over the stage's
  fixed-width UTC datetime text (see ``date_trunc_text``).

Units: second, minute, hour, day, week (starting Sunday), week_monday, month,
quarter, year, week_ending_saturday and week_ending_sunday (the last day of a week
that starts on Sunday or Monday respectively).
"""

import datetime
from typing import Any, Dict, Optional

UNITS = (
    "second",
    "minute",
    "hour",
    "day",
    "week",
    "week_monday",
    "month",
    "quarter",
    "year",
    "week_ending_saturday",
    "week_ending_sunday",
)
_TEXT = "%Y-%m-%d %H:%M:%S.%f"


def _check(unit: str) -> str:
    unit = unit.lower()
    if unit not in UNITS:
        raise ValueError(f"Unsupported DATE_TRUNC unit: {unit!r}")
    return unit


def mongo_expression(unit: str, field: str) -> Dict[str, Any]:
    """The aggregation expression for DATE_TRUNC(unit, field)."""
    unit = _check(unit)
    date = f"${field}"
    if unit in ("week", "week_ending_saturday"):
        truncated = {"$dateTrunc": {"date": date, "unit": "week", "startOfWeek": "sunday"}}
    elif unit in ("week_monday", "week_ending_sunday"):
        truncated = {"$dateTrunc": {"date": date, "unit": "week", "startOfWeek": "monday"}}
    else:
        truncated = {"$dateTrunc": {"date": date, "unit": unit}}
    if unit.startswith("week_ending"):
        return {"$dateAdd": {"startDate": truncated, "unit": "day", "amount": 6}}
    return truncated


def truncate(unit: str, value: datetime.datetime) -> datetime.datetime:
    unit = _check(unit)
    if unit == "second":
        return value.replace(microsecond=0)
    if unit == "minute":
        return value.replace(second=0, microsecond=0)
    if unit == "hour":
        return value.replace(minute=0, second=0, microsecond=0)
    day = value.replace(hour=0, minute=0, second=0, microsecond=0)
    if unit == "day":
        return day
    if unit in ("week", "week_ending_saturday"):
        start = day - datetime.timedelta(days=(day.weekday() + 1) % 7)  # back to Sunday
    elif unit in ("week_monday", "week_ending_sunday"):
        start = day - datetime.timedelta(days=day.weekday())  # back to Monday
    elif unit == "month":
        return day.replace(day=1)
    elif unit == "quarter":
        return day.replace(month=(day.month - 1) // 3 * 3 + 1, day=1)
    else:  # year
        return day.replace(month=1, day=1)
    return start + datetime.timedelta(days=6) if unit.startswith("week_ending") else start


def datetime_text(value: Any) -> Optional[str]:
    """Fixed-width UTC text for a datetime (or ISO string) value."""
    if value is None:
        return None
    if isinstance(value, str):
        value = datetime.datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    if isinstance(value, datetime.datetime):
        if value.tzinfo is not None:
            value = value.astimezone(datetime.timezone.utc).replace(tzinfo=None)
        return value.strftime(_TEXT)
    if isinstance(value, datetime.date):
        return datetime.datetime(value.year, value.month, value.day).strftime(_TEXT)
    raise ValueError(f"Not a datetime: {value!r}")


def date_trunc_text(unit: str, value: Any) -> Optional[str]:
    """SQLite function DATE_TRUNC(unit, datetime text)."""
    text = datetime_text(value)
    if text is None:
        return None
    return truncate(unit, datetime.datetime.strptime(text, _TEXT)).strftime(_TEXT)


def str_to_datetime_text(*args: Any) -> Optional[str]:
    """SQLite function STR_TO_DATETIME(text[, format]), as the value function of the same name."""
    from ..sql.value_function_registry import ValueFunctionRegistry

    if args and args[0] is None:
        return None
    return datetime_text(ValueFunctionRegistry.str_to_datetime(*args))


def datetime_positions(sql: str) -> list:
    """Positions of the outer projections that are DATE_TRUNC/STR_TO_DATETIME calls.

    Their SQLite result is datetime text (an expression has no declared type); the
    caller converts it back to datetime.
    """
    lowered = sql.lower()
    if "date_trunc" not in lowered and "str_to_datetime" not in lowered:
        return []
    try:
        import sqlglot
        from sqlglot import exp

        tree = sqlglot.parse_one(sql, read="sqlite")
    except Exception:
        return []
    if not isinstance(tree, exp.Select):
        return []
    positions = []
    for index, projection in enumerate(tree.expressions):
        inner = projection.this if isinstance(projection, exp.Alias) else projection
        name = inner.name.lower() if isinstance(inner, exp.Anonymous) else ""
        if isinstance(inner, exp.DateTrunc) or name in ("date_trunc", "str_to_datetime"):
            positions.append(index)
    return positions


def parse_text(value: Any) -> Any:
    return datetime.datetime.strptime(value, _TEXT) if isinstance(value, str) else value
