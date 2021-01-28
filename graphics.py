"""
Graphics Module.

This module provides utility classes for graphing information.  It ensures
both that data can be easily visualized and that all such visualizations
follow similar aesthetic principles.

The heart of the module is the LivePlot class.  It is initialized with a
set of Plotters and maintains and updates them continuously.  Each Plotter
is in turn responsible for handling a single data subplot.
"""

# TODO flesh out the typing here

from typing import Any, Dict, List, Optional, Tuple

from abc import abstractmethod

import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

class Plotter:
    """
    Abstract class providing an API for plotting data.
    """

    @abstractmethod
    def attach(self, axis):
        """
        Attach this plotter to a plot axis
        """

    @abstractmethod
    def on_update(self):
        """
        Signifies data may have been changed and the corresponding plot should
        be updated.
        """

class LivePlot:
    """
    This class handles a single "live" plot consisting of a variable number of
    subplots.  It accepts a series of standardized plotting methods - one per
    subplot - and performs the necessary setup.  To update the plots, simply
    call "update()".
    """
    # This does one thing
    # pylint: disable=too-few-public-methods

    plotters: List[Plotter]
    shape: Tuple[int, int]
    args: List[Dict[str, Any]]

    def __init__(self, plotters: List[Plotter], *,
                 shape: Optional[Tuple[int, int]] = None):
        """
        Init function.  Accepts a list of +plotter+ functions and a 2D +shape+
        in which to organize their graphs.
        """

        # If shape isn't provided, generate it
        if not shape:
            shape = (1, len(plotters))

        # Sanity check
        if len(plotters) != shape[0] * shape[1]:
            raise RuntimeError("Incorrect shape provided")

        self.plotters = plotters
        self.shape = shape

        # Ensure interactive mode is on
        plt.ion()

        # Set up style
        sns.set(style="darkgrid")
        plt.style.use("dark_background")

        # Generate figure and axes.  Note that this logic is complex solely
        # because plt.subplots() returns tuples of various sizes and nesting
        # topologies which makes dereferencing them a complete nightmare.
        # Whoever designed that system, know that God hates you.
        self.figure, axis_tuples = plt.subplots(nrows=shape[0], ncols=shape[1])
        axes = []
        if shape[0] == 1:
            if shape[1] == 1:
                axes = [axis_tuples]
            else:
                axes = [*axis_tuples]
        else:
            if shape[1] == 1:
                axes = [*axis_tuples]
            else:
                for subtuple in axis_tuples:
                    axes.extend([*subtuple])

        # Attach plotters
        for index, plotter in enumerate(plotters):
            plotter.attach(axes[index])

        # Finally, display the plot
        self.figure.tight_layout()
        mng = plt.get_current_fig_manager()
        mng.window.showMaximized()
        plt.show()

    def update(self):
        """
        Updates all data.
        """
        # Ensure interactive mode is on
        plt.ion()

        # Run update on each subplot
        for plotter in self.plotters:
            plotter.on_update()

        # Let the plotter catch up
        plt.pause(0.5)


class ListPlotter(Plotter):
    """
    Class which plots simple lists of values.
    """

    def __init__(self, data: List[List[Any]], x_label: str = '', y_label: str = '',
                 title: str = ''):
        """
        Init function.  Takes a list of +data+ to plot and various optional graph
        metainformation.
        """
        self.data = data
        self.x_label = x_label
        self.y_label = y_label
        self.title = title

        # Variable for remebering the lines whenever they needs updating
        self._lines = []

    def attach(self, axis):
        """
        Attach this plotter to a plot axis
        """
        # Set title and labels
        axis.set_title(self.title)
        axis.set_xlabel(self.x_label)
        axis.set_ylabel(self.y_label)

        # Plot data and remember the line for future updates
        for line in self.data:
            self._lines.append(axis.plot(range(len(line)), line)[0])

    def on_update(self):
        """
        Signifies data may have been changed and the corresponding plot should
        be updated.
        """
        # Reaffirm data on plot in case it has changed
        for index, line in enumerate(self._lines):
            line.set_xdata(range(len(self.data[index])))
            line.set_ydata(self.data[index])


class DataFramePlotter(Plotter):
    """
    Class which plots data from a pandas DataFrame and plots it in a standard
    way.  Updates to the +frame+ will cause the plot to be updated on the next
    cycle.
    """

    frame: pd.DataFrame

    def __init__(self, frame: pd.DataFrame, *, x_var: Optional[str] = None,
                 y_vars: Optional[List[str]] = None, **kwargs: Dict[str, str]):
        """
        Init function.

        Plots a data +frame+.  Uses +x_var+ as the index of the x-axis variable
        and draws a line for each variable in +y_vars+.  The x label, y label,
        and title are given by +x_label+, +y_label+, and +title+, respectively.

        Note that a None +x_var+ defaults to the index.
        """
        super().__init__()
        self.frame = frame
        self.x_var = x_var
        self.y_vars = y_vars
        self.axis = None
        self.kwargs = kwargs

        self._lines = {}

        # Sanity check parameters

        if not y_vars or not [var for var in y_vars if var in frame]:
            raise RuntimeError(f"Invalid y-axis variables: '{y_vars}'")

    def attach(self, axis):
        """
        Attach this plotter to a plot axis
        """
        self.axis = axis

        # Set title and labels
        if 'title' in self.kwargs:
            axis.set_title(self.kwargs['title'])
        if 'x_label' in self.kwargs:
            axis.set_xlabel(self.kwargs['x_label'])
        if 'y_label' in self.kwargs:
            axis.set_ylabel(self.kwargs['y_label'])

        # Perform custom formatting
        self._format(axis)

        # Plot each variable and remember the lines for future updates
        x_series = self.frame[self.x_var] if self.x_var else self.frame.index
        for y_var in self.y_vars:
            self._lines[y_var] = axis.plot(x_series, self.frame[y_var],
                                           label=y_var)[0]
        axis.legend()

    def _format(self, axis) -> None:
        """
        Handles formatting using passed arguments.
        """
        # First, handle time axis formatting if requested
        if self.kwargs.get('time_format', '') in ['decade', 'year', 'day']:
            fmt = self.kwargs['time_format']
            if fmt == 'decade':
                major_locator = mpl.dates.YearLocator(2)
                minor_locator = mpl.dates.MonthLocator()
                major_formatter = mpl.dates.DateFormatter('%Y')
            elif fmt == 'year':
                major_locator = mpl.dates.MonthLocator()
                minor_locator = mpl.dates.WeekdayLocator(byweekday=mpl.dates.MO)
                major_formatter = mpl.dates.DateFormatter('%m')
            elif fmt == 'day':
                major_locator = mpl.dates.HourLocator()
                minor_locator = mpl.dates.MinuteLocator(15)
                major_formatter = mpl.dates.DateFormatter('%H:%M')
            axis.xaxis.set_major_locator(major_locator)
            axis.xaxis.set_major_formatter(major_formatter)
            axis.xaxis.set_minor_locator(minor_locator)
            plt.setp(axis.get_xticklabels(), rotation='30', ha='right')

    def on_update(self) -> None:
        """
        Signifies data may have been changed and the corresponding plot should
        be updated.
        """
        # Run update on each line in case the data has changed
        x_series = self.frame[self.x_var] if self.x_var else self.frame.index
        for var, line in self._lines.items():
            line.set_xdata(x_series)
            line.set_ydata(self.frame[var])

        # TODO add data width limiting logic
        #if np.min(y1_data)<=line1.axes.get_ylim()[0] or np.max(y1_data)>=line1.axes.get_ylim()[1]:
        #plt.ylim([np.min(y1_data)-np.std(y1_data),np.max(y1_data)+np.std(y1_data)])


class AVDataPlotter(DataFramePlotter):
    """
    Class which plots data from AlphaVantage.
    """

    def __init__(self, data: pd.DataFrame, ticker: str, magnitude: str):
        """
        Plots a +data+ frame representing a particular +magnitude+ of time for
        a given +ticker+ symbol.
        """

        # Get time data
        if magnitude == "minute":
            title = f"{ticker} @ {data.index[0].strftime('%d/%m/%Y')}"
            fmt = "day"
        elif magnitude == "day":
            title = f"{ticker} @ {data.index[0].year}"
            fmt = "year"
        elif magnitude == "week":
            title = f"{ticker} Weekly"
            fmt = "decade"
        elif magnitude == "month":
            title = f"{ticker} Monthly"
            fmt = "decade"
        else:
            raise RuntimeError(f"Incorrect time magnitude provided: {magnitude}")

        super().__init__(data, y_vars=["high", "low"],
                         x_label="Time", y_label="Price", title=title,
                         time_format=fmt)


def sleep(seconds: int) -> None:
    """
    Sleep for +seconds+ seconds.  Serves as a replacement for time.sleep()
    which doesn't block graphical processes.
    """
    plt.pause(seconds)
