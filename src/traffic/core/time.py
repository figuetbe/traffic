import logging
from datetime import datetime, timezone  # noqa: F401
from numbers import Real

from pyopensky.time import (
    deltalike,
    time_or_delta,
    timelike,
    timetuple,
    to_timedelta,
)

import pandas as pd

__all__ = [
    "deltalike",
    "time_or_delta",
    "timelike",
    "timetuple",
    "to_datetime",
    "to_timedelta",
]

_log = logging.getLogger(__name__)


def to_datetime(time: timelike) -> pd.Timestamp:
    """Facility to convert anything to a pd.Timestamp.

    >>> f"{to_datetime('2017-01-14')}"
    '2017-01-14 00:00:00+00:00'
    >>> f"{to_datetime('2017-01-14 12:00Z')}"
    '2017-01-14 12:00:00+00:00'
    >>> f"{to_datetime(1484395200)}"
    '2017-01-14 12:00:00+00:00'
    >>> f"{to_datetime(datetime(2017, 1, 14, 12, tzinfo=timezone.utc))}"
    '2017-01-14 12:00:00+00:00'
    """

    if isinstance(time, str):
        time = pd.Timestamp(time, tz="utc")
    if isinstance(time, datetime):
        time = pd.Timestamp(time)
    if isinstance(time, Real):
        time = pd.Timestamp(float(time), tz="utc", unit="s")
    if time.tzinfo is None:  # coverage: ignore
        _log.warning(
            "This timestamp is tz-naive. Things may not work as expected. "
            "If you construct your timestamps manually, consider passing a "
            "string, which defaults to UTC. If you construct your timestamps "
            "automatically, look at the tzinfo (resp. tz) argument of the "
            "datetime (resp. pd.Timestamp) constructor."
        )
    return time
