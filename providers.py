"""
Market Data Provider.

This module contains implementations of the DataProvider abstract class, which
defines methods by which market information can be requested and presented.
"""

from abc import abstractmethod
from io import StringIO
import os
import pathlib
import time
from typing import Any, Dict

import pandas as pd
import requests


class DataProvider:
    """
    Abstract class defining the DataProvider API.
    """

    @abstractmethod
    def intraday(self, day: pd.Timestamp):
        """
        Gets the intraday data for a given day.
        """

    @abstractmethod
    def daily(self, year: pd.Timestamp):
        """
        Gets the yearly data for a given +year+.
        """

    @abstractmethod
    def weekly(self):
        """
        Returns a frame containing all weekly data+.
        """

    @abstractmethod
    def monthly(self):
        """
        Returns a frame containing all monthly data.
        """

    @abstractmethod
    def first(self) -> pd.Timestamp:
        """
        Returns the earliest date for which all types of data are available.
        """

    @abstractmethod
    def latest(self) -> pd.Timestamp:
        """
        Returns the latest date for which all types of data are available.
        """

    def access_all(self):
        """
        Simulates accesses of all kinds.  Designed to allow caching
        implementations to perform all of their caching up front.
        """


class AVDataProvider(DataProvider):
    """
    An implementation of DataProvider which uses the AlphaVantage API.
    """

    def __init__(self, ticker: str, *,
                 reqs_per_minute: int = 5, cache: str = "cache",
                 local_cache_size: int = 10,
                 **kwargs: Dict[str, Any]):
        """
        Init function.

        +reqs_per_minute+ is the number of requests allowed per minute.
        +ticker+ provides the ticker symbol for the underlying FD.
        +cache+ provides a directory which the DataProvider can use to
        organize data.
        +local_cache_size+ is the total number of entries to keep on-hand to
        speed up repeated accesses.

        NOTE: This object assumes it is the only user of the API key at any
        given time, and will attempt the maximum number of accesses possible.
        """
        self.ticker = ticker
        self.reqs_per_minute = reqs_per_minute
        self.cache = pathlib.Path(cache)
        self.local_cache_size = local_cache_size

        self._calls = []
        self._local_cache = {}
        self._local_cache_history = []

        # Ensure the cache is suitable
        if self.cache.exists() and not self.cache.is_dir():
            raise RuntimeError("Cache must be a directory")
        self.cache.mkdir(exist_ok=True, parents=True)

        # Get AlphaVantage API key
        self.api_key = os.environ.get("SKINTBROKER_AV_API_KEY")
        if not self.api_key:
            raise RuntimeError("No AlphaVantage API key detected - please set "
                               "SKINTBROKER_AV_API_KEY")

    def _check_local_cache(self, filename: pathlib.Path):
        """
        Checks for data associated with a given +filename+ in the local cache.
        If found, return it, else return None.
        """
        if str(filename) in self._local_cache:
            cache_entry = self._local_cache[str(filename)]
            if len(self._local_cache) == self.local_cache_size:
                self._local_cache_history.remove(str(filename))
                self._local_cache_history.append(str(filename))
            return cache_entry
        return None

    def _add_local_cache(self, filename: pathlib.Path, frame: pd.DataFrame):
        """
        Adds a +frame+ associated with a given +filename+ to the local cache.
        If the cache is full, pops off the least recently accessed entry.
        """
        # If necessary, purge the oldest item from the cache
        if len(self._local_cache) == self.local_cache_size:
            old_name = self._local_cache_history.pop(0)
            del self._local_cache[old_name]
        self._local_cache[str(filename)] = frame
        self._local_cache_history.append(str(filename))

    def intraday(self, day: pd.Timestamp):
        """
        Gets the intraday data for a given day.
        """
        # TODO handle today data

        # First, check if the data is already cached
        cache_dir = self.cache/self.ticker/str(day.year)/str(day.month)
        csv = cache_dir/f"{day.day}_per_minute.csv"
        data = self._check_local_cache(csv)
        if data is not None:
            return data
        if cache_dir.exists() and csv.exists():
            frame = pd.read_csv(csv, parse_dates=[0],
                                infer_datetime_format=True,
                                index_col='time')
            self._add_local_cache(csv, frame)
            return frame

        # Otherwise, download it.  Intraday data is divided into 30-day
        # segments, so first determine just how far back to look.
        days = (_now().floor('d') - day.floor('d')).days - 1
        month = (days // 30) % 12 + 1
        year = (days // 360) + 1
        params = {"function": "TIME_SERIES_INTRADAY_EXTENDED",
                  "interval": "1min",
                  "symbol": self.ticker,
                  "slice": f"year{year}month{month}"}
        request_frame = self._api_request(**params)
        if request_frame.empty:
            return None

        # Cache all downloaded data - no point in wasting queries!
        grouper = pd.Grouper(freq='D')
        for date, group in request_frame.groupby(grouper):
            date_dir = self.cache/self.ticker/str(date.year)/str(date.month)
            date_csv = date_dir/f"{date.day}_per_minute.csv"
            if not date_csv.exists():
                date_dir.mkdir(exist_ok=True, parents=True)
                group.to_csv(date_csv, index_label='time')

        # Try again.  If there's still no data, there probably isn't any.
        if csv.exists():
            frame = pd.read_csv(csv, parse_dates=[0],
                                infer_datetime_format=True,
                                index_col='time')
            self._add_local_cache(csv, frame)
            return frame

        return None

    def daily(self, year: pd.Timestamp):
        """
        Gets the yearly data for a given +year+.
        """
        # First, check if the data is already cached
        now = _now()
        cache_dir = self.cache/self.ticker/str(year.year)
        csv = cache_dir/"per_day.csv"
        data = self._check_local_cache(csv)
        if data is not None:
            return data
        if cache_dir.exists() and csv.exists():
            frame = pd.read_csv(csv, parse_dates=[0],
                                infer_datetime_format=True,
                                index_col='time')

            # If the data is from this year yet it isn't today's data,
            # update anyway.
            if year.year != now.year or \
               frame.index[0].dayofyear != now.dayofyear:
                self._add_local_cache(csv, frame)
                return frame

        # Update from remote
        params = {"function": "TIME_SERIES_DAILY_ADJUSTED",
                  "symbol": self.ticker,
                  "outputsize": "full"}
        request_frame = self._api_request(**params)

        # Cache all returned data
        grouper = pd.Grouper(freq='Y')
        for date, group in request_frame.groupby(grouper):
            date_dir = self.cache/self.ticker/str(date.year)
            date_csv = date_dir/"per_day.csv"

            # If the CSV is missing OR it's this year, then cache
            if not date_csv.exists() or date.year == now.year:
                date_dir.mkdir(exist_ok=True, parents=True)
                group.to_csv(date_csv, index_label='time')

        # Try again.  If there's still no data, there probably isn't any.
        if csv.exists():
            frame = pd.read_csv(csv, parse_dates=[0],
                                infer_datetime_format=True,
                                index_col='time')
            self._add_local_cache(csv, frame)
            return frame
        return None

    def weekly(self):
        """
        Returns a frame containing all weekly data.
        """
        # First, check if the data is already cached
        now = _now()
        cache_dir = self.cache/self.ticker
        csv = cache_dir/"per_week.csv"
        data = self._check_local_cache(csv)
        if data is not None:
            return data
        if cache_dir.exists() and csv.exists():
            frame = pd.read_csv(csv, parse_dates=[0],
                                infer_datetime_format=True,
                                index_col='time')

            # If the data isn't recent, update
            if frame.index[0].week == now.week:
                self._add_local_cache(csv, frame)
                return frame

        # Update from remote
        # Set up call parameters
        params = {"function": "TIME_SERIES_WEEKLY_ADJUSTED",
                  "symbol": self.ticker}
        request_frame = self._api_request(**params)

        # Cache returned data.
        if not cache_dir.exists():
            cache_dir.mkdir(exist_ok=True, parents=True)
        request_frame.to_csv(csv, index_label='time')

        # Try again.  If there's still no data, there probably isn't any.
        if csv.exists():
            frame = pd.read_csv(csv, parse_dates=[0],
                                infer_datetime_format=True,
                                index_col='time')
            self._add_local_cache(csv, frame)
            return frame
        return None

    def monthly(self):
        """
        Returns a frame containing all monthly data.
        """
        # First, check if the data is already cached
        now = _now()
        cache_dir = self.cache/self.ticker
        csv = cache_dir/"per_month.csv"
        data = self._check_local_cache(csv)
        if data is not None:
            return data
        if cache_dir.exists() and csv.exists():
            frame = pd.read_csv(csv, parse_dates=[0],
                                infer_datetime_format=True,
                                index_col='time')

            # If the data isn't recent, update
            if frame.index[0].month == now.month:
                self._add_local_cache(csv, frame)
                return frame

        # Update from remote
        # Set up call parameters
        params = {"function": "TIME_SERIES_MONTHLY_ADJUSTED",
                  "symbol": self.ticker}
        request_frame = self._api_request(**params)

        # Cache returned data.
        if not cache_dir.exists():
            cache_dir.mkdir(exist_ok=True, parents=True)
        request_frame.to_csv(csv, index_label='time')

        # Try again.  If there's still no data, there probably isn't any.
        if csv.exists():
            frame = pd.read_csv(csv, parse_dates=[0],
                                infer_datetime_format=True,
                                index_col='time')
            self._add_local_cache(csv, frame)
            return frame
        return None

    def _api_request(self, **kwargs: Dict[str, str]) -> pd.DataFrame:
        """
        Performs an API request using the passed parameters.  Returns a
        DataFrame or None.
        """

        # Assemble the query
        site = "https://www.alphavantage.co/query?"
        params = [f"{key}={val}" for key, val in \
                {**kwargs, "apikey": self.api_key, "datatype": "csv"}.items()]
        query = "&".join(params)

        # Perform call limit bookkeeping, and delay if needed.
        if len(self._calls) >= self.reqs_per_minute:
            oldest_call = self._calls.pop(0)
            to_wait = 60 - (_now() - oldest_call).seconds
            if to_wait >= 0:
                time.sleep(to_wait + 1)

        # Call the API and generate the dataframe
        print("Querying: " + site + query)
        response = requests.get(site + query)
        response.encoding = 'utf-8'
        index_label = 'time' if "INTRADAY" in kwargs["function"] \
                      else 'timestamp'

        frame = pd.read_csv(StringIO(response.text), parse_dates=[0],
                            infer_datetime_format=True,
                            index_col=index_label)

        # Record this call for future checks
        self._calls.append(_now())

        return frame

    def first(self) -> pd.Timestamp:
        """
        Returns the earliest date for which all types of data are available.
        """
        # Based on the AlphaVantage system, it's reasonable to assume data
        # exists for two years back from today.  Note that it's entirely
        # possible that cached data exists from even earlier, so a future
        # extension should search for it.
        return _now() - pd.Timedelta(720 - 1, unit='d')

    def latest(self) -> pd.Timestamp:
        """
        Returns the latest date for which all types of data are available.
        """
        # Yesterday is fine
        return _now() - pd.Timedelta(1, unit='d')

    def access_all(self) -> None:
        """
        Simulates accesses of all kinds.  Designed to allow caching
        implementations to perform all of their caching up front.
        """
        # First, handle daily, weekly, and monthly entries for the last 20
        # years.  As this comes in one immense blob, just access that.
        now = _now()
        self.monthly()
        self.weekly()
        self.daily(now)

        # Then handle intraday for the last 2 years.
        days = pd.date_range(end=now, freq='D', periods=360 * 2 - 1)
        for day in days:
            if day.weekday() <= 4:
                self.intraday(day)


def _now() -> pd.Timestamp:
    """
    Returns the current DateTime.
    """
    return pd.to_datetime("now")
