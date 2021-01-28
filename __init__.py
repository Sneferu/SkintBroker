"""
SkintBroker.

This package provides a framework for implementing, training, and using
neural nets to predict trends in US equities markets.  Each module provides
services associated with one aspect of this process.

The 'providers' module provides implementations of the DataProvider class,
which is responsible for generating pandas DataFrames containing market info
across a specified range of dates and at a specified granularity.

The 'training' module provides methods of generating training and test data
using raw data from one or more providers.

The 'graphics' module provides utility functions for plotting data, both
from the market and from the nets themselves.

The 'models' module provides implementations of various machine learning
architectures on which to train the data.

Finally, the 'utils' module contains utility functions which don't fit
cleanly into anything else.
"""
