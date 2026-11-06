"""
TEST FIXTURE ONLY. An approximate NYSE calendar so the clock logic can be
tested without WRDS. Real runs take trading days from crsp.dsf_v2
(common.build_trading_calendar).
"""
import datetime as dt

import pandas as pd
from pandas.tseries.holiday import GoodFriday
from pandas.tseries.offsets import DateOffset, WeekOfMonth, LastWeekOfMonth  # noqa: F401


def nth_weekday(year, month, weekday, n):
    d = dt.date(year, month, 1)
    d += dt.timedelta(days=(weekday - d.weekday()) % 7)
    return d + dt.timedelta(weeks=n - 1)


def last_weekday(year, month, weekday):
    d = dt.date(year, month + 1, 1) - dt.timedelta(days=1)
    return d - dt.timedelta(days=(d.weekday() - weekday) % 7)


def observed(d, saturday_rule=True):
    if d.weekday() == 5:
        return d - dt.timedelta(days=1) if saturday_rule else None
    if d.weekday() == 6:
        return d + dt.timedelta(days=1)
    return d


def holidays(year):
    h = [
        observed(dt.date(year, 1, 1), saturday_rule=False),
        nth_weekday(year, 1, 0, 3), nth_weekday(year, 2, 0, 3),
        last_weekday(year, 5, 0), observed(dt.date(year, 7, 4)),
        nth_weekday(year, 9, 0, 1), nth_weekday(year, 11, 3, 4),
        observed(dt.date(year, 12, 25)),
    ]
    if year >= 2022:
        h.append(observed(dt.date(year, 6, 19)))
    h += [d.date() for d in GoodFriday.dates(f"{year}-01-01", f"{year}-12-31")]
    return {d for d in h if d is not None}


SPECIAL_CLOSED = {dt.date(2007, 1, 2), dt.date(2012, 10, 29), dt.date(2012, 10, 30),
                  dt.date(2018, 12, 5), dt.date(2025, 1, 9)}


def make(start=2005, end=2026):
    hol = set().union(*(holidays(y) for y in range(start, end + 1))) | SPECIAL_CLOSED
    days = [d.date() for d in pd.bdate_range(f"{start}-01-01", f"{end}-12-31") if d.date() not in hol]
    cal = pd.DataFrame({"date": pd.to_datetime(days), "close_et": "16:00"})
    early = set()
    for y in range(start, end + 1):
        early.add(nth_weekday(y, 11, 3, 4) + dt.timedelta(days=1))   # day after Thanksgiving
        early.add(dt.date(y, 12, 24))
        early.add(dt.date(y, 7, 3))
    cal.loc[cal["date"].dt.date.isin(early), "close_et"] = "13:00"
    return cal


if __name__ == "__main__":
    cal = make()
    cal.to_csv("trading_calendar_TEST_ONLY.csv", index=False)
    print(len(cal), "days written")
