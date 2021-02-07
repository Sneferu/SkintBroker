"""
Market Data Presenter.

This module contains implementations of the DataPresenter abstract class, which
is responsible for presenting data in the form of mxnet tensors.  Each
implementation presents a different subset of the available data, allowing
different models to make use of similar data.
"""
from typing import Dict, List, Optional, Tuple

from abc import abstractmethod

import pandas as pd
import numpy as np
from mxnet import ndarray as nd

from . import providers, utils

class DataPresenter:
    """
    Abstract class defining the DataProvider API.
    """

    @abstractmethod
    def get_training_batch(self, size: int):
        """
        Returns a batch of training data, partitioned from the validation data,
        of size +size+.
        """

    @abstractmethod
    def get_validation_batch(self, size: int):
        """
        Returns a batch of validation data, partitioned from the training data,
        of size +size+.
        """

    @abstractmethod
    def data_array(self, timestamp: pd.Timestamp):
        """
        Returns the data associated with a single +timestamp+ in mxnet form
        """

    @abstractmethod
    def data_frame(self, timestamp: pd.Timestamp):
        """
        Returns the data associated with a single +timestamp+ in pandas form.
        """

    @abstractmethod
    def data_features(self) -> List[str]:
        """
        Returns a list of data features in the same order as presented in the
        frames.
        """


class IntradayPresenter:
    """
    Loads data consisting only of intraday information, guaranteed to keep all
    within market hours.
    """
    # All it does is load data - no other calls necessary
    # pylint: disable=too-few-public-methods

    def __init__(self, provider: providers.DataProvider, *, window: int = 45,
                 valid_seed: int = 0, lookahead: int = 10,
                 normalize: bool = True, features: Dict[str, bool] = {},
                 **kwargs):
        """
        Init function.  Takes a +provider+ from which it extracts data and
        a variety of other arguments.  See info files for examples.
        """
        # pylint: disable=too-many-instance-attributes

        # Store basic setup parameters
        self.provider = provider
        self._window = window
        self._valid_seed = valid_seed
        self._lookahead = lookahead
        self._normalize = normalize
        self._features = [feat for feat in features if features[feat]]
        self._outputs = []

        # Collect and decide features
        for feature in self._features:
            # First handle special features
            if feature == 'macd':
                self._outputs.append('macd_signal')
            if feature == 'vortex':
                self._outputs.extend(['vortex+', 'vortex-'])
                continue
            if feature == 'stochastic':
                self._outputs.extend(['%K', '%D'])
                continue
            if feature == 'williams':
                self._outputs.append('%R')
                continue

            # Then add all others
            self._outputs.append(feature)

        # Decide range of possible dates in advance
        self._first = provider.first()
        # TODO don't limit this anymore
        self._latest = provider.latest() - pd.to_timedelta(2, unit='day')

        # Cache for already processed data to cut down on disk usage
        self._train_cache = {}
        self._val_cache = {}

        # Cache of holidays to prevent continuously recalculating them
        self._holidays = utils.trading_holidays(self._first - pd.to_timedelta(1, unit='day'),
                                                self._latest)
        self._half_days = utils.trading_half_days(self._first - pd.to_timedelta(1, unit='day'),
                                                  self._latest)

    def get_training_batch(self, size: int) -> Tuple[nd.NDArray, nd.NDArray]:
        """
        Returns a batch of training data, partitioned from the validation data,
        of size +size+.
        """
        return self._get_batch(size, validation=False)

    def get_validation_batch(self, size: int) -> Tuple[nd.NDArray, nd.NDArray]:
        """
        Returns a batch of validation data, partitioned from the training data,
        of size +size+.
        """
        return self._get_batch(size, validation=True)

    def data_array(self, timestamp: pd.Timestamp) -> nd.NDArray:
        """
        Returns the data associated with a single +timestamp+ in mxnet form
        """
        start_time = timestamp - pd.to_timedelta(self._window, unit='min')
        return self._get_data(start_time, False)[0]

    @abstractmethod
    def data_frame(self, timestamp: pd.Timestamp):
        """
        Returns the data associated with a single +timestamp+ in pandas form.
        """
        data =  self._extract_daily_data(timestamp)
        if data is None:
            return None
        return data.loc[timestamp, :]

    def _get_data(self, time: pd.Timestamp, validation: bool) \
            -> Tuple[nd.NDArray, nd.NDArray]:
        """
        Returns a simgle data sample starting at a given +time+.  Uses
        +validation+ to distinguish between training and validation sets.

        NOTE: This function assumes that the entire data window is available.
              If a time provided is too late to obtain a full window, behavior
              is UNPREDICTABLE.
        """
        # Check if the sample has already been cached.
        day = time.floor('D')
        start_index = (time.hour - 9) * 60 + (time.minute - 30)
        end_index = start_index + self._window
        if validation and day in self._val_cache:
            data, target = self._val_cache[day]
            return data[start_index: end_index], target[start_index: end_index]
        if not validation and day in self._train_cache:
            data, target = self._train_cache[day]
            return data[start_index: end_index], target[start_index: end_index]

        # Otherwase generate, cache, and return it
        data, target = self._to_daily_input_data(day)
        if validation:
            self._val_cache[day] = (data, target)
        else:
            self._train_cache[day] = (data, target)

        return data[start_index: end_index], target[start_index: end_index]

    def _to_daily_input_data(self, date: pd.Timestamp) \
            -> Tuple[nd.NDArray, nd.NDArray]:
        """
        Transforms a set of intraday data for a +date+ to an array appropriate
        for input to the model, and a target set of predictions against which
        to compare outputs.
        """
        # Gather data requested data components.  Note that this seemingly
        # over-complicated method guarantees that they remain in the order
        # prescribed by the feature list.
        datas = []
        for feat in self._outputs:
            if feat == "high":
                datas.append(_to_intraday_high(date, self.provider,
                                               normalize=self._normalize))
            elif feat == "low":
                datas.append(_to_intraday_low(date, self.provider,
                                              normalize=self._normalize))
            elif feat == "change":
                datas.append(_to_intraday_change(date, self.provider,
                                                 normalize=self._normalize))
            elif feat == "open":
                datas.append(_to_intraday_open(date, self.provider,
                                               normalize=self._normalize))
            elif feat == "volume":
                datas.append(_to_intraday_volume(date, self.provider,
                                                 normalize=self._normalize))
            elif feat == "time":
                datas.append(_to_intraday_time(date, self.provider,
                                               normalize=self._normalize))
            elif feat == "macd":
                # For MACD, include both MACD and its signal
                macd, macd_signal = _to_intraday_macd(date, self.provider,
                                                      normalize=self._normalize)
                datas.extend([macd_signal, macd])
            elif feat == "mass_index":
                datas.append(_to_intraday_mass_index(date, self.provider))
            elif feat == "trix15":
                datas.append(_to_intraday_trix(date, self.provider, 15))
            elif feat == "vortex+":
                vortex_up, vortex_down = _to_intraday_vortex(date,
                                                             self.provider, 25)
                datas.extend([vortex_up, vortex_down])
            elif feat == "%K":
                pK, pD = _to_intraday_stochastic(date, self.provider, 30)
                datas.extend([pK, pD])
            elif feat == "rsi":
                datas.append(_to_intraday_rsi(date, self.provider, 14))
            elif feat == "%R":
                # The Williams %R is mathematically equivalent to (1 - %K). It
                # is duplicated here to obtain a shorter period.
                pK, _ = _to_intraday_stochastic(date, self.provider, 10)
                datas.append(pK - 1)
            elif feat == "accdist":
                datas.append(_to_intraday_accdist(date, self.provider))
            elif feat == "mfi":
                datas.append(_to_intraday_mfi(date, self.provider, 30))
            elif feat == "vpt":
                datas.append(_to_intraday_vpt(date, self.provider))
            elif feat == "target":
                datas.append(_to_intraday_target(date, self.provider,
                                                 self._lookahead,
                                                 normalize=self._normalize))

        # Gather target data and return data/target arrays
        target = _to_intraday_target(date, self.provider, self._lookahead,
                                     normalize=self._normalize)

        return nd.stack(*datas, axis=1), target.reshape(-1, 1)

    def _extract_daily_data(self, date: pd.Timestamp) -> Optional[pd.DataFrame]:
        """
        Gets the market data for a given day, restricted to market hours.
        """
        data = self.provider.intraday(date)
        if data is None or data.empty:
            return None
        return data

    def _get_batch(self, batch_size: int, validation: bool = False) \
            -> Tuple[nd.NDArray, nd.NDArray]:
        """
        Gets a random batch of data of size +batch_size+.  Returns a tuple of
        data and target predictions.  If +validation+ is set, prevents these
        dates from being drawn for non-validation batches.
        """

        # Define a Callable for testing appropriate dates
        def _is_suitable_time(time: pd.Timestamp) -> bool:
            """
            Returns whether the market is open at a given +time+ for the
            required window.
            """

            # First, confirm that this date matches the right type
            day = time.floor(freq='D')
            is_validation_date = (day.dayofyear % 10 == self._valid_seed)
            if validation != is_validation_date:
                return False

            # Ensure it's on weekdays and during market hours.  Note that we
            # discard the last 10 minutes of trading because they are both
            # dangerous for day trading and provide no good way to train the
            # 10 minute output for the model.
            if time.weekday() > 4:
                return False
            if (time.hour * 60 + time.minute) < 9 * 60 + 30:
                return False
            if (time.hour * 60 + time.minute + self._window) > 15 * 60 - self._lookahead:
                return False

            # Check aginst holidays.  Note that for the sake of sanity, we
            # don't include half days.
            if day in self._holidays or day in self._half_days:
                return False

            return True

        # Next, generate arrays of random dates within the last two years,
        # recording appropriate ones to form an array of size +batch_size+
        timestamps = pd.Series()
        while True:
            random_times = pd.to_datetime(np.random.randint(low=self._first.value,
                                                            high=self._latest.value,
                                                            size=(100),
                                                            dtype='int64')).to_series()
            suitable_mask = random_times.apply(_is_suitable_time)
            timestamps = pd.concat([timestamps, random_times.loc[suitable_mask]])
            if len(timestamps) >= batch_size:
                timestamps = timestamps[0 : batch_size]
                break
        index_array = pd.to_datetime(timestamps)

        # Next, gather all data into batches with axes (batch, window, data...)
        datas, targets = [], []
        for timestamp in index_array:
            data, target = self._get_data(timestamp, validation)
            datas.append(data)
            targets.append(target)
        data_array, target_array = nd.stack(*datas), nd.stack(*targets)

        # Return the data
        return data_array, target_array

    def data_features(self) -> List[str]:
        """
        Returns a list of data features in the same order as presented in the
        frames.
        """
        return self._outputs


def _get_intraday_data(date: pd.Timestamp, provider: providers.DataProvider) \
        -> pd.DataFrame:
    """
    Gets the intraday datafrome limited to market hours for a given +date+
    and +provider+.
    """
    # First, get data and limit it to market hours
    data = provider.intraday(date)
    if data is None or data.empty:
        raise RuntimeError(f"Something went wrong - empty data array for {date}!")
    start = data.index[0].replace(hour=9, minute=30)
    end = data.index[0].replace(hour=16, minute=0)

    # Next, resample the data by the minute and interpolate missing values
    data = data.loc[data.index.isin(pd.date_range(start=start, end=end, freq='min'))]
    data = data.resample('min')
    data = data.interpolate(method='time').copy()

    return data

def _to_intraday_high(date: pd.Timestamp, provider: providers.DataProvider,
                      normalize: bool = True) -> nd.NDArray:
    """
    Returns an ndarray consisting of the per-minute high of a data series for
    a given +date+ and +provider+.  If +normalize+, it is divided by the
    open price.
    """
    data = _get_intraday_data(date, provider)
    high = ((data.high - data.open) / data.open) if normalize else data.high
    return nd.array(high.values, utils.try_gpu(0))

def _to_intraday_low(date: pd.Timestamp, provider: providers.DataProvider,
                      normalize: bool = True) -> nd.NDArray:
    """
    Returns an ndarray consisting of the per-minute high of a data series for
    a given +date+ and +provider+.  If +normalize+, it is divided by the
    open price.
    """
    data = _get_intraday_data(date, provider)
    low = ((data.low - data.open) / data.open) if normalize else data.low
    return nd.array(low.values, utils.try_gpu(0))

def _to_intraday_change(date: pd.Timestamp, provider: providers.DataProvider,
                        normalize: bool = True) -> nd.NDArray:
    """
    Returns an ndarray consisting of the per-minute close of a data series for
    a given +date+ and +provider+.  If +normalize+, it is divided by the
    previous close
    """
    data = _get_intraday_data(date, provider)
    close_prev = data.close.shift(periods=1, fill_value=data.close[0])
    close = ((data.close - close_prev) / close_prev) if normalize else data.close
    return nd.array(close.values, utils.try_gpu(0))

def _to_intraday_open(date: pd.Timestamp, provider: providers.DataProvider,
                      normalize: bool = True) -> nd.NDArray:
    """
    Returns an ndarray consisting of the per-minute open of a data series for
    a given +date+ and +provider+.  If +normalize+, it is divided by the
    daily open price.
    """
    data = _get_intraday_data(date, provider)
    open = (data.open / data.open.iloc[0]) if normalize else data.open
    return nd.array(open.values, utils.try_gpu(0))

def _to_intraday_volume(date: pd.Timestamp, provider: providers.DataProvider,
                        normalize: bool = True) -> nd.NDArray:
    """
    Returns an ndarray consisting of the per-minute high of a data series for
    a given +date+ and +provider+.  If +normalize+, it is divided by the
    average volume.
    """
    data = _get_intraday_data(date, provider)
    vol = data.volume / data.volume.mean() if normalize else data.volume
    return nd.array(vol.values, utils.try_gpu(0))

def _to_intraday_time(date: pd.Timestamp, provider: providers.DataProvider,
                      normalize: bool = True) -> nd.NDArray:
    """
    Returns an ndarray consisting of the trading minute of a data series for
    a given +date+ and +provider+.  If +normalize+, it is normalized so that
    9:30 is 0 and 16:00 is 1
    """
    data = _get_intraday_data(date, provider)
    minute = data.index.hour * 60 + data.index.minute - (9 * 60 + 30)
    tempus = (minute / (60 * 7 + 30)) if normalize else minute
    return nd.array(tempus.values, utils.try_gpu(0))

def _to_intraday_macd(date: pd.Timestamp, provider: providers.DataProvider,
                      normalize: bool = True) -> Tuple[nd.NDArray, nd.NDArray]:
    """
    Returns a pair of ndarrays consisting of the per-minute MACD of a data
    series for a given +date+ and +provider+, and a signal for the same.  If
    normalize+, both are divided by the daily open price.
    """
    # First, calculate the MACD via exponential moving averages
    data = _get_intraday_data(date, provider)
    ewm12 = pd.Series.ewm(data['close'], span=12).mean()
    ewm26 = pd.Series.ewm(data['close'], span=26).mean()
    macd = ewm26 - ewm12

    # Next, calculate the signal line
    signal = pd.Series.ewm(macd, span=9).mean()

    # Return both
    return nd.array(macd.values, utils.try_gpu(0)), \
           nd.array(signal.values, utils.try_gpu(0))

def _to_intraday_trix(date: pd.Timestamp, provider: providers.DataProvider,
                      period: int)-> Tuple[nd.NDArray, nd.NDArray]:
    """
    Returns an ndarray containing the TRIX for a given +data+ and +provider+,
    averaged across a given +period+.
    """
    # First, get the triple-smoothed 15 period exponential moving average
    data = _get_intraday_data(date, provider)
    ewm1 = pd.Series.ewm(data['close'], span=period).mean()
    ewm2 = pd.Series.ewm(ewm1, span=period).mean()
    ewm3 = pd.Series.ewm(ewm2, span=period).mean()

    # Return the percentage change from last period
    ewm3_yesterday = ewm3.shift(periods=1, fill_value=ewm3[0])
    trix = (ewm3 / ewm3_yesterday) - 1
    return nd.array(trix.values, utils.try_gpu(0))

def _to_intraday_mass_index(date: pd.Timestamp,
                            provider: providers.DataProvider) -> nd.NDArray:
    """
    Returns an ndarray consisting of the per-minute mass index of a data series
    for a given +date+ and +provider+.
    """
    # First, calculate the difference between high and low
    data = _get_intraday_data(date, provider)
    diff = data['high'] - data['low']

    # Next, calculate the moving average of the difference (and its moving
    # average)
    ewm9 = pd.Series.ewm(diff, span=9).mean()
    ewm99 = pd.Series.ewm(ewm9, span=9).mean()
    ratio = ewm9/ewm99

    # Sum and return
    mass_index = ratio.rolling(25).sum()/250
    return nd.array(mass_index.values, utils.try_gpu(0))

def _to_intraday_vortex(date: pd.Timestamp, provider: providers.DataProvider,
                        period: int) -> Tuple[nd.NDArray, nd.NDArray]:
    """
    Returns a pair of ndarrays consisting of the positive and negative vortex
    indicators of a data series for a given +date+ and +provider+, taken over
    a given +period+.
    """
    # First, calculate the True Range
    data = _get_intraday_data(date, provider)
    prev_close = data.close.shift(periods=1, fill_value=data.close[0])
    high_low = (data.high - data.low).abs()
    low_close = (data.low - prev_close).abs()
    high_close = (data.high - prev_close).abs()
    true_range = pd.concat([high_low, low_close, high_close], axis=1).min(axis=1)

    # Next, calculate the vortex moements
    prev_low = data.low.shift(periods=1, fill_value=data.low[0])
    prev_high = data.high.shift(periods=1, fill_value=data.high[0])
    vm_up = (data.high - prev_low).abs()
    vm_down = (data.low - prev_high).abs()

    # Finally, calculate the indicator itself
    true_range_sum = true_range.rolling(period).sum()
    vm_up_sum = vm_up.rolling(period).sum()
    vm_down_sum = vm_down.rolling(period).sum()
    vi_up = vm_up_sum / true_range_sum
    vi_down = vm_down_sum / true_range_sum

    # Return VI+ and VI- values
    return nd.array(vi_up.values, utils.try_gpu(0)), \
           nd.array(vi_down.values, utils.try_gpu(0))

def _to_intraday_rsi(date: pd.Timestamp, provider: providers.DataProvider,
                     period: int) -> nd.NDArray:
    """
    Returns an ndarray consisting of the per-minute Relative Strength Index of
    a data series for a given +date+ and +provider+ over a given +period+.
    """
    # First, calculate the per-period up and down movements
    data = _get_intraday_data(date, provider)
    prev_close = data.close.shift(periods=1, fill_value=data.close[0])
    movement = data.close - prev_close
    move_up = movement.where(movement > 0, 0)
    move_down = -movement.where(movement < 0, 0)

    # Calculate the relative strength and relative strength index
    ewm_up = pd.Series.ewm(move_up, span=period).mean()
    ewm_down = pd.Series.ewm(move_down, span=period).mean()
    rs = ewm_up/ewm_down
    rsi = 1 - 1/(1 + rs)
    return nd.array(rsi.values, utils.try_gpu(0))

def _to_intraday_stochastic(date: pd.Timestamp, provider: providers.DataProvider,
                            period: int) -> Tuple[nd.NDArray, nd.NDArray]:
    """
    Returns a pair of ndarrays consisting of the %K and %D values associated
    with the stochastic oscillator framework for a given +date+ and +provider+,
    taken over a given +period+.
    """
    # First, get the total highs and lows over the previous 30 time periods.
    data = _get_intraday_data(date, provider)
    high = data.high.rolling(period).max()
    low = data.low.rolling(period).min()

    # Next, calculate the %K and %D
    pK = (data.close - low) / (high - low)
    pD = pK.rolling(3).mean()

    # Return them
    return nd.array(pK.values, utils.try_gpu(0)), \
           nd.array(pD.values, utils.try_gpu(0))

def _to_intraday_accdist(date: pd.Timestamp, provider: providers.DataProvider) \
        -> nd.NDArray:
    """
    Returns an ndarray consisting of the per-minute Accumulation/Distribution
    Index of a data series for a given +date+ and +provider+.
    """
    # First, get the data
    data = _get_intraday_data(date, provider)

    # Next, calculate the Current Money Flow Volume
    cmfv = (2 * data.close) - (data.high + data.low)
    cmfv *= (data.volume / 1000) / (0.0001 + data.high - data.low)

    # Now generate the Acc/Dist index for each timestamp
    accdist = np.empty(len(cmfv))
    accdist[0] = cmfv.iloc[0]
    for i in range(1, len(cmfv)):
        accdist[i] = accdist[i - 1] + cmfv.iloc[i]

    # Return the Acc/Dist index
    return nd.array(accdist / np.linalg.norm(accdist), utils.try_gpu(0))

def _to_intraday_mfi(date: pd.Timestamp, provider: providers.DataProvider,
                     period: int) -> nd.NDArray:
    """
    Returns an ndarray consisting of the per-minute Money Flow Index of a data
    series for a given +date+ and +provider+ accross a given +period+.
    """
    # First, get the data
    data = _get_intraday_data(date, provider)

    # Next, calculate the typical price and money_flow
    typical_price = (data.high + data.low + data.close) / 3
    money_flow = typical_price * data.volume

    # Find the positive and negative money flows
    prev_typical_price = typical_price.shift(periods=1,
                                             fill_value=typical_price[0])
    positive_flow = money_flow.where(typical_price > prev_typical_price, 0)
    negative_flow = money_flow.where(typical_price < prev_typical_price, 0)

    # Sum over the window and return the ratio
    positive = positive_flow.rolling(period).sum()
    negative = negative_flow.rolling(period).sum()
    mfi = positive / (positive + negative)
    return nd.array(mfi.values, utils.try_gpu(0))

def _to_intraday_vpt(date: pd.Timestamp, provider: providers.DataProvider) \
        -> nd.NDArray:
    """
    Returns an ndarray consisting of the per-minute Volume Price Trend of a
    data series for a given +date+ and +provider+.
    """
    # First, get the data
    data = _get_intraday_data(date, provider)

    # Next, multiply the change by the volume
    prev_close = data.close.shift(periods=1, fill_value=data.close[0])
    vpt = data.volume * (data.close - prev_close) / prev_close

    # Return the VPT
    return nd.array(vpt.values, utils.try_gpu(0))

def _to_intraday_target(date: pd.Timestamp, provider: providers.DataProvider,
                        offset: int, normalize: bool = True) -> nd.NDArray:
    """
    Returns an ndarray consisting of the target values of a data series for
    a given +date+ and +provider+, offset by +offset+ minutes.  If +normalize+,
    it is divided by the per-minute open.
    """
    data = _get_intraday_data(date, provider)
    close = data.close.shift(periods=-offset, fill_value=0)
    target = ((close - data.open) / data.open) if normalize else close
    return nd.array(target.values, utils.try_gpu(0))
