# SkintBroker

Market Models Managed Right

## Purpose

Market modeling as a science is almost as old as markets themselves.  Technology has evolved from ex avibus augury through hand-drawn charts to modern quantitative trading, wherein supercomputers fight each other mere feet from stock exchanges, and billion dollar bets are placed against five cent, five second swings.  Against such powerful models, what hope does the retail trader have?  Enter SkintBroker.

SkintBroker is a system for designing and deploying models to predict medium term market actions.  It aims to circumvent the mighty institutional traders by hunting in the no man's land between their camps - actions slow enough to be ignored by microsecond arbitrage bots but fast enough that technical analysts just shrug.  In practice, these motions are on the scale of a few minutes to a couple hours - the average attention span of an amateur options trader.  To this intended audience, SkintBroker can serve as a powerful weapon for competing against professional firms and - hopefully - give them a fighting chance at beating the market.

## Usage

SkintBroker supports two operating modes.  In the first - command line - it takes a series of arguments, such as a stock ticker, a timeframe, and an action, and outputs data in either textual form or as a set of graphs.  In the second - package form - it offers its library of models and technical indicators to other python projects.

### Setup

First things first, setup.  SkintBroker works pretty much out of the box, as long as you have the right modules installed.  These are:

__mxnet 1.7.0__ (GPU Enabled Preferred)\
__matplotlib 3.3.4__\
__pandas 1.2.2__\
__requests 2.25.1__\
__PyYAML-5.4.1__

I haven't tested newer versions of these packages, though SkintBroker doesn't leverage many arcane features outside of MXNet, so there shouldn't be too many problems upgrading.  It is, however, highly dependent on the ever-evolving MXNet package, so alter that version at your own risk.

Additionally, make sure you have a couple GB of disk space and a directory with rwx permissions for data caching.  More on this later.

### Command Line

Skintbroker can be called like any other python module with a ```__main__,py``` - via the ```python -m ...``` command.  It has three major subcommands associated with it: data, model, and compare.  There is a set of flags applicable to all of these commands, as well as several associated specifically with each.

#### Global Flags

SkintBroker supports four global flags applicable to any subcommand.  These specify blueprint, item, ticker, and cache.

__-f \<blueprint_file\>__: This flag specifies a blueprint yaml file.  These are located in the blueprints/ subdirectory, and consist of a series of specifications detailing how to construct a model.  They are discussed further in the Blueprints section

__-i \<item\>__: An individual model inside a blueprint file.  As several models can overlap, it is useful to define them within one yaml file.  This flag selects one my name.  If it is omitted, the entire file is treated as one unnamed model.

__-s \<ticker\>__: This is the ticker for the stock we want to model.

__-c \<cache_dir\>__: This is the place where training data is cached.  Note that this flag is designed to be as broad as possible, leaving the internal structure of the cache to the data provider.

#### Data

If models are the heart of SkintBroker, data is the lifeblood.  The ability to gather and supply such data is extremely important, but only slightly less so is the ability to view it and confirm its reality.  To the effect, SkintBroker provides a robust set of commands for gathering and displaying data through use of the '__data__' command, which in turn splits into '__data cache__' and '__data show__'.

The '__data cache__' command uses the provided ticker and cache\_dir flags and begins data collection, going back as far as the underlying provider allows, and storing it in the cache.  Note that some providers may be rate-limited by the actual data source (such as a stock data website), and will take a long time to download and store everything.  The data provider should continously print its progress, and will exit when caching is complete.  Note that caching is itself simply a proactive measure, as data downloaded for model training will be cached through the same mechanism.

The '__data show__' command comes with the '__-t \<timestamp\>__' flag, which takes a date and displays the high and low data for that date, that week, that month, and that year.

Examples:

    python -m Skintbroker -c ~/my\_cache -s SPY -f providers/av data cache
    # Caches all data avaiable to the default AlphaVantage provider for $SPY.  This may take some time.
    # See the Providers section for info on how to use the default provider.

    python -m Skintbroker -c ~/my\_cache -s WFC -f provider/av data show -t 8/23/2021
    # Displays the price movements for $WFC on 8/23/2021

#### Model

Models are the heart of SkintBroker.  There are two phases of model use - training and prediction.  Consequently, there are two subcommands.

The '__model train__' command trains a model.  It takes the following flags:

__-e \<epochs\>__: The number of times to cycle through the training data.  For neural-net based models, this is the familiar epochs parameter.

__-r__: The 'reinit' flag.  Indicates that the model's parameters should be reinitialized before training.  Exclude it if you want to keep honing a previously trained model.

The '__model predict__' command takes one flag: __-t \<timestamp\>__.  This flag takes a date, and the framework will run the model on a minute by minute basis, making a prediction for some distance into the future (default 10 minutes).  If the date specified is today, it will predict up to the current minute.

Examples:

    python -m Skintbroker -c ~/my\_cache -s SPY -f technical-all -i fibonacci model predict -t 8/23/2021
    # Uses the Fibonacci Retracement technical indicator in the technical-all.yaml blueprint file to make
    # 10 minute predictions for 8/23/2021.

    python -m Skintbroker -c ~/my\_cache -s WFC -f neural-simple -i textcnn model train -e 100 -r:
    # Reinitializes the parameters for the textcnn convolutional neural network model and retrains it for
    # 100 epochs.  When this is done, the 'model train' command can be used to make predictions with it.

#### Compare

The '__compare__' subcommand is fairly straightforward.  It prints out a list of success rates for all models which have been initialized (i.e. put through at least one training cycle).  It provides a good gauge for which techniques are working and which are not.

#### Blueprints

The Blueprints system provides a way to define model parameters in a reusable way.  It consists of a set of yaml files full of dictionaries, along with the !include extension which allows one file's dictionaries to be included in another.  The premise is fairly straightforward - each model is defined in terms of a data 'provider', a data 'presenter', and the 'model' itself.  These terms are defined more broadly in the Design section, but essentially, the provider collects the data, the presenter formats it and feeds it into the model, and the model is trained and performs predictions.  A typical blueprint is laid out below.  Please check the Design section if any of its components seem confusing.

```
model-name:
    provider:
        type: # The provider type
        params: # A parameter dictionary to be fed to the provider
    presenter:
        type: # The presenter type
        params: # The presenter parameters
            features: # A list of the features to include.  These features are checked by the models that use them.
            ...: # Any other set of type-specific parameters
    nets: # The model itself
        net1: # One of the sub-models (nets) in the model
            type: # The type of net.  Tells the framework which class to use
            inputs: # The inputs to this net.  It is either <presenter> or another net.
            params: # A parameter dictionary to feed the model.  Unique to every net type.
        ... # An arbitrary set of nets can be listed.  The framework connects them through their 'inputs' lists
```

To run this model, use the __-f__ flag to point to this file along with '__-i model-name__'.  Note that when performing data operations (e.g. show or cache), point to a provider definition instead.

The blueprint files themselves make heavy use of yaml references and anchors.  For more info on yaml syntax, check out one of many online tutorials, or visit https://yaml.org/spec/1.2/.

### Package

_Independent package support is still ongoing and will probably have to wait for the next tag.  In theory the classes detailed in the Design section can be imported on their own, but I have yet to test them thoroughly.  Anyone who wants to contribute, please try this out._

### Providers

The actual data for training and using the model is provided by a Provider.  While we defer a discussion of its internals until the Design section, suffice it to say that the Provider is responsible for gathering raw data from a reliable source, caching it efficiently, and forwarding it to the model.

Currently, the default provider gathers data from AlphaVantage.  To use it, first go to www.alphavantage.co and get an API key.  Next, edit the parameters in the AVProvider blueprint - they are set by default to work with the free API key, but if you're willing to pay more for faster accesses, you'll want to adjust them accordingly.  Finally, make sure to set the ```SKINTBROKER_AV_API_KEY``` environment variable to your API key.  This system avoids placing it in the blueprint file and committing it by accident.

## Design

It is not the strongest of the models that survives; it is the one most adaptable to change.  With market dynamics ever shifting, and more powerful theories to explain them developed every year, it is imperative that no market prediction framework remain static.  SkintBroker is designed to be as extensible as possible, making the addition of new models as simple as extending a class.  This section covers the design philosophy behind SkintBroker, and goes into some detail on how to build and incorporate new models.

### The Big Picture

The SkintBroker framework consists of three main components and the logic to glue them together.  These are Providers, Presenters, and Models.

The Provider is responsible for gathering data.  It is responsible for interfacing with an external source, performing any necessary caching, and providing blocks of standard-format data with standard timesteps.

The Presenter is responsible for transforming the standardized data and converting it into something the model can understand.  Usually this takes two phases - first, perform any transforms which would be difficult or redundant to perform within the model (such as calculating RSI), and second, convert it into an MXNet block.  This block is then fed - with a list of contained features - to the model.

The Model is the heavy lifter, the real heart and soul of SkintBroker.  It takes an MXNet array and outputs an MXNet array, which either simply feeds into another model, or performs a prediction by outputting an array of market sentiments - the percentage certainty that the market will go up, down, or sideways within the specified window.  A model can be as simple as a traditional market indicator, though the best models consist of both traditional indicators and neural nets, stacked like legos into complex block structures.

The most important part of the glue logic (besides the framework which performs training and prediction) is the loss system.  During the training phase, the final output of a model (the market sentiments mentioned above) is compared with the actual market performance, and a loss is calculated.  Available are the standard losses (L1, L2, etc), along with a special loss called "Gambling Loss".  Given a market sentiment, this loss assumes that a trader bets percentages of his portfolio on up movements and down movements proportionally to the up and down sentiments.  For example, if ```(up, down, sideways)``` is ```(0.3, 0.2, 0.5)```, it assumes the trader bets 30% on up and 20% on down, with sideways serving as a catchall for the excess probability so the sum of all three sentiments is always 1.  The loss then calculates how much money the trader would lose (or, if negative, make).  It tends to be more accurate than traditional losses for market applications, and a gradient-friendly approximation (the GFGLoss) is implemented for training neural nets.

Below are sections on how to extend Providers, Presenters, and Models so that the framework will recognize them.  Note that right now, this system is rather dirty - requiring direct modifications of the code - but the project is still in alpha, and a better solution will be forthcoming in the future.

### Adding Models

Adding a model is fairly simple.  Simple extend the ```Net``` class in ```models/model.py```.  This class in turn extends the MXNet Block class, allowing it to be easily incorporated into the MXNet framework, but adds a couple Skintbroker specific methods to override:


```features()```: This is the list of features output by this net.  It is useful when it feeds another net, so that mismatches in the feature set can be immediately detected.

```trainable()```: A boolean function, determines whether there is any use trying to train this net, or if it should be ignored during any training phase.  A neural net is trainable, a technical indicator - probably not.

```begin_state()```: Any initialization to perform before training/prediction.

Additionally, it requires a list of features to be fed into ```__init__()```.  These are the features it expects from the net or provider which feeds it, and it will check for them at construction time.

### Adding Providers

All providers extend the ```DataProvider``` class in ```providers.py```, and must implement the following functions:

```intraday()```:
Provides a pandas dataframe of minute-by-minute timestamped data for a requested day.  Must include open, high, low, and close.

```daily()```: Provides a pandas dataframe of day-by-day timestamped data for a requested year.  Must include open, high, low, and close.

```weekly()```: Provides a pandas dataframe of week-by-week timestamped data for all available time.  Must include open, high, low, and close.

```monthly()```: Provides a pandas dataframe of month-by-month timestamped data for all available time.  Must include open, high, low, and close.

```first()```: Returns the earliest date for which all above types of data are available

```latest()```: Returns the latest date for which all aboe types of data are available

```access_all()```: Simulates accesses of all data for all available dates.  This function is designed to allow the framework to request immediate caching of all data which could be requested for training or prediction.  If the Provider doesn't perform caching - or if the implementer is feeling lazy - this function can be empty.  Note that if this function could perform caching but doesn't, the Provider implementation is unlikely to get upstream merge approval.

### Adding Presenters

Adding presenters, while possible, is somewhat redundant.  The ```IntradayPresenter``` in ```presenters.py``` has all of the logic for adding a number of features.  Simply follow that example - add a function to generate your feature, add a check in ```IntradayPresenter``` to select it, then make sure you request it in your blueprint file.

## Improvements

SkintBroker is still very much an alpha stage project.  As such, it has a lot of room for improvements.  Here are some things in the pipeline:

* Conversion into a PyPI package (without mucking up the command line implementation too much)
* Implementation of new models/presenters/providers without modifying existing code
* Providers outside AlphaVantage
* SQL-based replacement for the ad-hoc data caching system
* More complicated NN models reflecting newer theory
* Support for longer term predictions/data windows
* General cleanup for outdated docstrings
* Linting support (tricky due to the highly convoluted nature of pandas data types)
* More technical indicators (God Willing - the 0.3-alpha update has quite a few already)

If you have any ideas - or want to contribute to the project - don't hesitate to contact me.  My support for this project goes in fits and starts based on my external schedule and any help would be greatly appreciated.

## Conclusion

If you made it through all of the above, you have the patience to make SkintBroker work for you.  Best of luck and happy trading!

Mac Scanlan

