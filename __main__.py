"""
Main File

This file contains the main() function used when running this package as a
program.  It allows access to all existing functionality through a well-
structured command line interface.
"""


from typing import Any, Dict, List

import argparse
import os
import pathlib

import pandas as pd

from . import constructor, parser, graphics


def record_success(cache: pathlib.Path, name: str, success: float) -> None:
    """
    Records the +success+ of a model +name+ in a CSV in the +cache+ directory.
    """
    storage = get_success_record(cache)
    storage.loc[name] = success
    storage.to_csv(cache/"success.csv", index_label='model')

def get_success_record(cache: pathlib.Path) -> pd.Series():
    """
    Gets the most recent success record in a +cache+ directory.
    """
    storage = cache/"success.csv"
    if not storage.exists():
        storage = pd.DataFrame(columns=['model', 'success'])
        storage.set_index('model', inplace=True)
    else:
        storage = pd.read_csv(storage, index_col=0)
    return storage

def parse_input(args) -> Dict[str, Any]:
    """
    Returns the dictionary from an info file based on +args+.
    """
    info_dir = pathlib.Path(os.path.dirname(os.path.realpath(__file__)))/"blueprints"
    return parser.parse_file(info_dir/f"{args.info_file}.yaml")[args.item]

def main() -> None:
    """
    Main function.
    """

    # Parse arguments
    arg_parser = argparse.ArgumentParser(prog="skintbroker",
                                         description="Market Data manipulation, analysis, and prediction tool")
    arg_parser.add_argument('-f', dest='info_file', help="Name of an info file to use",
                            type=str, default="")
    arg_parser.add_argument('-i', dest='item', help="Name of a relevant item in the info file",
                            type=str, default="")
    arg_parser.add_argument('-s', dest='ticker', type=str, help="Ticker symbol")
    arg_parser.add_argument('-c', dest='cache_dir', type=pathlib.Path, help="Cache directory")

    # Data commands
    subparsers = arg_parser.add_subparsers(dest='command')
    data_parser = subparsers.add_parser('data', help="Manipulate market data")
    data_subparsers = data_parser.add_subparsers(dest='subcommand')
    show_parser = data_subparsers.add_parser('show', help="Show market data")
    show_parser.add_argument('-t', dest='timestamp', type=str, help="Timestamp")
    data_subparsers.add_parser('cache', help="Download and cache market data")

    # Model commands
    model_parser = subparsers.add_parser('model', help="Train or predict with a model")
    model_subparsers = model_parser.add_subparsers(dest='subcommand')
    train_parser = model_subparsers.add_parser("train", help="Train a model")
    train_parser.add_argument('-e', dest='train_epochs', type=int, help="Epochs to train")
    train_parser.add_argument('-r', dest='reinit', action='store_true',
                              help="Reinitialize parameters")
    pred_parser = model_subparsers.add_parser("predict", help="Predict via a model")
    pred_parser.add_argument('-t', dest='timestamp', type=str, help="Timestamp")

    # Comparison commands
    subparsers.add_parser("compare", help="Compare all recorded models")

    args = arg_parser.parse_args()

    # Base the next decision on the command given
    if args.command == 'data':
        # Parse info file and generate the provider
        item_info = parse_input(args)
        provider = constructor.build_provider(args.ticker, item_info)
        if args.subcommand == 'cache':

            # Download all data available for a ticker
            provider.access_all()

        elif args.subcommand == 'show':
            # Show cached data
            timestamp = pd.Timestamp(args.timestamp)

            # Gather data
            data_intra = provider.intraday(timestamp)
            data_daily = provider.daily(timestamp)
            data_weekly = provider.weekly()
            data_monthly = provider.monthly()

            # Plot it
            plot_intra = graphics.AVDataPlotter(data_intra, args.ticker, "minute")
            plot_daily = graphics.AVDataPlotter(data_daily, args.ticker, "day")
            plot_weekly = graphics.AVDataPlotter(data_weekly, args.ticker, "week")
            plot_month = graphics.AVDataPlotter(data_monthly, args.ticker, "month")
            liveplot = graphics.LivePlot([plot_month, plot_daily, plot_weekly,
                                          plot_intra], shape=(2, 2))
            while True:
                liveplot.update()

    elif args.command == 'model':
        # Parse info file and generate the model
        item_info = parse_input(args)
        model = constructor.build_model(args.ticker, args.item, item_info)
        model_cache = args.cache_dir/args.ticker/"models"

        if args.subcommand == 'train':
            model.initialize(save=model_cache, random_init=args.reinit)

            # Train the model for the specified number of epochs
            training_loss, valid_loss, valid_success = model.train(args.train_epochs)

            # Save the model
            model.save(model_cache)
            if len(valid_success) > 3:
                success = sum(valid_success[-3:])/3
            else:
                success = valid_success[-1]
            record_success(model_cache, args.item, success)

            # Finally, generate the plot
            title = f"Training loss per epoch for {args.item}"
            plotter1 = graphics.ListPlotter([training_loss],
                                            x_label="Epoch",
                                            y_label="Training Loss",
                                            title=title)
            plotter2 = graphics.ListPlotter([valid_loss, valid_success],
                                            x_label="Epoch",
                                            y_label="Valid loss/success",
                                            title="Valid loss/success per epoch")
            liveplot = graphics.LivePlot([plotter1, plotter2], shape=(1, 2))

            while True:
                liveplot.update()
                graphics.sleep(10)

        elif args.subcommand == 'predict':
            model.initialize(save=model_cache, random_init=False)

            # Predict and plot, rerunning each cycle if the data has changed
            plotter1, plotter2, liveplot = None, None, None
            while True:

                # Generate time range
                # TODO
                start = pd.to_datetime(args.timestamp).replace(hour=10, minute=15)
                end = start.replace(hour=15, minute=50)
                times = pd.date_range(start=start, end=end, freq='min')
                prediction = None
                for time in times:
                    # Generate prediction
                    pred = model.predict(time)
                    if prediction is None:
                        prediction = pred
                    else:
                        prediction = prediction.append(pred)
                if model.output_type() == "prediction":
                    prediction['10min'] = prediction['10min'].shift(periods=10)

                # Generate or update plotter
                if plotter1:
                    plotter1.frame = prediction
                    if plotter2:
                        plotter2.frame = prediction
                    continue

                if model.output_type() == "sentiment":
                    title1 = "Market price"
                    title2 = "Predicted market sentiment"
                    y_vars1 = ["open"]
                    y_vars2 = ["up", "down", "side"]
                    plotter1 = graphics.DataFramePlotter(prediction,
                                                         y_vars=y_vars1,
                                                         x_label="Time",
                                                         y_label="Price",
                                                         title=title1,
                                                         time_format='day')
                    plotter2 = graphics.DataFramePlotter(prediction,
                                                         y_vars=y_vars2,
                                                         x_label="Time",
                                                         y_label="Sentiment",
                                                         title=title2,
                                                         time_format='day')
                    liveplot = graphics.LivePlot([plotter1, plotter2],
                                                 shape=(2, 1))

                elif args.output == "prediction":
                    title = "Market Prices (Real and Predicted)"
                    y_vars = ["open", "10min"]
                    plotter1 = graphics.DataFramePlotter(prediction,
                                                         y_vars=y_vars,
                                                         x_label="Time",
                                                         y_label="Price",
                                                         title=title,
                                                         time_format='day')
                    plotter2 = None
                    liveplot = graphics.LivePlot([plotter1], shape=(1, 1))

                # Wait 30 seconds to prevent unnecessary reruns
                liveplot.update()
                graphics.sleep(30)

    elif args.command == 'compare':
        model_cache = args.cache_dir/args.ticker/"models"
        success_rates = get_success_record(model_cache)
        print(success_rates.sort_values(by="success"))

if __name__ == "__main__":
    main()
