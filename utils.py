"""
Utils

This module contains utility functions useful all over SkintBroker but not
cleanly fitting in any other module.
"""

import pandas as pd
from pandas.tseries.holiday import (AbstractHolidayCalendar, Day, Holiday,
                                    nearest_workday, USMartinLutherKingJr,
                                    USPresidentsDay, GoodFriday, USMemorialDay,
                                    USLaborDay, USThanksgivingDay)

import mxnet as mx


class USTradingHolidayCalendar(AbstractHolidayCalendar):
    """
    Pandas calendar for determining trading holidays
    """
    # This just adds a rule set.  All methods are contained in the superclass.
    # pylint: disable-too-few-public-methods
    rules = [Holiday("NewYearsDay", month=1, day=1,
                     observance=nearest_workday),
             USMartinLutherKingJr,
             USPresidentsDay,
             GoodFriday,
             USMemorialDay,
             Holiday("USIndependenceDay", month=7, day=4,
                     observance=nearest_workday),
             USLaborDay,
             USThanksgivingDay,
             Holiday("Christmas", month=12, day=25,
                     observance=nearest_workday)
            ]


class USTradingHalfDayCalendar(AbstractHolidayCalendar):
    # This just adds a rule set.  All methods are contained in the superclass.
    # pylint: disable=too-few-public-methods
    """
    Pandas calendar for determining trading half days.
    """
    rules = [Holiday("DayAfterThanksgivingDay", month=11, day=1,
                     offset=[USThanksgivingDay.offset, Day(1)]),
             Holiday("ChristmasEve", month=12, day=24),
             Holiday("IndependenceEve", month=7, day=3)
            ]


def trading_holidays(start: pd.Timestamp, end: pd.Timestamp):
    """
    Returns a list of trading holidays between +start+ and +end+ as a
    TimeSeries.
    """
    return USTradingHolidayCalendar().holidays(start, end)


def trading_half_days(start: pd.Timestamp, end: pd.Timestamp):
    """
    Returns a list of trading half days between +start+ and +end+ as a
    TimeSeries.
    """
    return USTradingHalfDayCalendar().holidays(start, end)


def get_gpu_count() -> int:
    """
    Returns the number of currently usable GPU's.
    """
    return 1
    # TODO update when 1.7.0 is available
    # return mx.context.num_gpus()


def try_gpu(number: int):
    """
    Returns a GPU context if that GPU exists.  Otherwise uses the CPU context.
    """
    if number >= get_gpu_count():
        return mx.cpu()
    return mx.gpu(number)
