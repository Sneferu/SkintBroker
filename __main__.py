# TODO header

from typing import List

import argparse
import pathlib

import pandas as pd

from .providers import DataProvider, AVDataProvider
from .presenters import DataPresenter, IntradayPresenter
from . import graphics, models


def select_model(args, features: List[str], presenters: List[DataPresenter]) \
        -> models.Model:
    """
    Returns a model for given sets of +args+ and +features+, presented by a
    list of +presenters+.
    """
    # Select loss, ensuring gambling loss is gradient-friendly
    loss_type = "gfg" if args.loss == "gambling" else args.loss
    loss = models.find_loss(loss_type, args.output)

    if args.model == 'rnn':
        return models.RecurrentModel(models.DailyRecurrentNet(args.output, features),
                                     presenters, window=45, verbose=True, loss=loss)

    elif args.model == 'conv':
        return models.Model(models.DailyConvolutionalNet(args.output, features),
                            presenters, window=45, verbose=True, loss=loss)

    elif args.model == 'textcnn':
        return models.Model(models.TextCNNNet(args.output, features),
                            presenters, window=45, verbose=True, loss=loss)

    elif "technical" in args.model:
        if len(args.model) > len("technical-"):
            top_model = args.model[len("technical-"):]
        else:
            top_model = None
        if len(args.features) == 1:
            singleton = args.features[0]
        elif not args.features and len(args.extras) == 1:
            singleton = args.extras[0]
        else:
            singleton = None
        return models.Model(models.TechnicalNet(args.output, features,
                                                singleton=singleton,
                                                extras=args.extras,
                                                top=top_model),
                            presenters, window=45, verbose=True, loss=loss)

    raise RuntimeError(f"No model {args.model} found")


def select_presenters(args, providers: List[DataProvider]) -> List[DataPresenter]:
    """
    Returns presenters for a given set of +args+ which get their data from a
    list of data +providers+.
    """
    # pylint: disable=no-else-return
    assert (0 <= args.valid_seed < 10), f"Invalid seed: {valid_seed}"

    # Set up extra feature list
    features = {feat: True for feat in args.features}

    # Default to 45 minute window
    return [IntradayPresenter(prov, 45, valid_seed=args.valid_seed, **features) \
            for prov in providers]


def get_model_cache(cache: str, ticker: str):
    """
    Gets the model parameter cache for a top level +cache+ directory and a
    +ticker+ symbol
    """
    return pathlib.Path(cache)/ticker/"models"


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


def main() -> None:
    """
    Main function.
    """
    # Big 'un
    # pylint: disable=too-many-statements,too-many-locals

    # TODO complete docstring

    # Define the list of possible features
    features = ['macd', 'mass_index', 'volume', 'trix15', 'vortex', 'rsi',
                'stochastic']

    # Parse arguments
    parser = argparse.ArgumentParser(prog="skintbroker",
                                     description="Market Data manipulation, analysis, and prediction tool")
    parser.add_argument('--api-key', dest='api_key', help="AlphaVantage API key")
    parser.add_argument('--cache', dest='cache', help="Market data cache")
    subparsers = parser.add_subparsers(dest='command')

    gather_parser = subparsers.add_parser('show', help="Gather market data")
    gather_parser.add_argument('-s', dest='ticker', type=str, help="Ticker symbol")
    gather_parser.add_argument('-t', dest='timestamp', type=str, help="Timestamp")

    cache_parser = subparsers.add_parser('cache', help="Cache all market data")
    cache_parser.add_argument('-s', dest='ticker', type=str, help="Ticker symbol")

    model_parser = subparsers.add_parser('compare', help="Compare models")
    model_parser.add_argument('-s', dest='ticker', type=str, help="Ticker symbol")

    model_parser = subparsers.add_parser('model', help="Run a model")
    model_parser.add_argument('-s', dest='ticker', type=str, help="Ticker symbol")
    model_parser.add_argument('-p', dest='timestamp', type=str, help='Time to predict')
    model_parser.add_argument('-t', dest='train_epochs', type=int, help='Epochs to train')
    model_parser.add_argument('-c', dest='compare', action="store_true",
                              help="Compare the success rates of different models")
    model_parser.add_argument('-m', dest='model', type=str,
                              help="The model to use")
    model_parser.add_argument('-f', dest='features', type=str,
                              choices=[*features, 'all', 'target'],
                              default=[], nargs='*', help="Extra features to include")
    model_parser.add_argument('-e', dest='extras', type=str,
                              choices=['slab', 'conv', 'textcnn'], default=[],
                              nargs='*', help="Extra arguments for the model")
    model_parser.add_argument('-l', dest='loss', type=str, choices=['l1', 'l2', 'gambling'],
                              default='l2', help="The loss to use")
    model_parser.add_argument('-o', dest='output', choices=['prediction', 'sentiment'],
                              default='prediction', help='The type of information output by the model')
    model_parser.add_argument('--valid-seed', dest='valid_seed', type=int, default=0,
                              help="An integer in [0, 10) used to select a validation set")
    model_parser.add_argument('--verbose', dest='verbose', action='store_true',
                              help="Activate verbose printing")
    model_parser.add_argument('--reinit', dest='reinit', action='store_true',
                              help="Reinitialize the model")

    args = parser.parse_args()

    # Modify arguments as necessary
    if 'features' in args and "all" in args.features:
        if 'target' in args.features:
            features.append('target')
        args.features = features

    # Create data provider
    provider = AVDataProvider(args.ticker, args.api_key, 5, pathlib.Path(args.cache))

    # Follow instructions
    if args.command == 'show':
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

    elif args.command == 'cache':

        # Download all data available for a ticker
        provider.access_all()

    elif args.command == 'model':

        # First, acquire and initialize a model
        presenters = select_presenters(args, [provider])
        model = select_model(args, presenters[0].data_features(), presenters)
        model_cache = get_model_cache(args.cache, args.ticker)
        cache = model_cache if not args.reinit else None
        model.initialize(save=cache, random_init=args.reinit)

        # Next, perform the required action
        if args.train_epochs:

            # Train for the specified number of epochs
            training_loss, valid_loss, valid_success = model.train(args.train_epochs)

            # Save the model
            model.save(model_cache)
            if len(valid_success) > 3:
                success = sum(valid_success[-3:])/3
            else:
                success = valid_success[-1]
            record_success(model_cache, model.name, success)

            # Finally, generate the plot
            title = f"Training loss per epoch for {str(args.model)}"
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

        if args.timestamp:

            # Predict and plot, rerunning each cycle if the data has
            # changed
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
                if args.output == "prediction":
                    prediction['10min'] = prediction['10min'].shift(periods=10)

                # Generate or update plotter
                if plotter1:
                    plotter1.frame = prediction
                    if plotter2:
                        plotter2.frame = prediction
                    continue

                if args.output == "sentiment":
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
            model_cache = get_model_cache(args.cache, args.ticker)
            success_rates = get_success_record(model_cache)
            print(success_rates.sort_values(by="success"))


if __name__ == "__main__":
    main()
