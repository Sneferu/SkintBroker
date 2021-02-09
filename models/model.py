
from typing import Any, Dict, List, Optional, Tuple

from abc import abstractmethod
import pathlib

import pandas as pd
import numpy as np
import mxnet as mx

from .. import presenters, utils

from .loss import find_loss


class Net(mx.gluon.Block):
    """
    Class representing an mxnet-style block, but with added properties.
    """

    def __init__(self, features: List[List[str]], **kwargs: Dict[str, Any]) \
            -> None:
        """
        Init function.

        +features+ is a list of input features for each input accepted.
        +inputs+ is a list of input array types accepted by the net.
        +output+ is a list of output items
        """
        super().__init__(**kwargs)
        self._input_features = features

    @property
    @abstractmethod
    def features(self) -> List[str]:
        """
        A list of features output by this net.
        """

    @property
    @abstractmethod
    def trainable(self) -> bool:
        """
        Whether or not this net is trainable
        """

    def begin_state(self, **kwargs: Dict[str, Any]) -> Any:
        """
        Performs state initialization.
        """
        return None


class Model:
    """
    Class for training and using a market model.
    """

    @abstractmethod
    def initialize(self, save: Optional[pathlib.Path] = None,
                   random_init: bool = True) -> None:
        """
        Initialize the model using either a set of saved parameters +save+,
        or random parameters if +random_init+ is set.
        """

    @abstractmethod
    def save(self, save: pathlib.Path):
        """
        Saves parameters to a file in the +save+ directory.
        """

    @abstractmethod
    def train(self, epochs: int) -> Tuple[List[float], List[float], List[float]]:
        """
        Trains the network for a number of epochs.  Returns the validation
        loss after each epoch, and the percentage-wise success.
        """

    @abstractmethod
    def predict(self, time: pd.Timestamp) -> pd.DataFrame:
        """
        Makes a prediction for a certain +time+.
        """

    @abstractmethod
    def output_type(self) -> str:
        """
        The output produced by this model.
        """


class SequentialModel:
    """
    Class implementing functions associated with training and using a
    sequential neural net.  Includes hooks designed for easy extension to
    recursive nets.
    """

    def __init__(self, net: Net, pres: List[presenters.DataPresenter],
                 name: str, *, window: int = 45, verbose: bool = False,
                 learning_rate: float = 0.0001, loss=mx.gluon.loss.L2Loss()) \
        -> None:
        """
        Initialization function.
        Ensures that this object gets an mxnet block +net+ and a set of data
        presenters +pres+ in decreasing priority in frames of size +window+.
        The +loss+ parameter allows selection of a training loss (default L2).
        The +verbose+ flag activates pseudo-debug printing.  The +name+
        parameter uniquely identifies the model for saving/loading.
        The +learning_rate+ is exactly what it sounds like.
        """
        self.net = net
        self.presenters = pres
        self.name = name
        self.window = window
        self.loss = loss
        self.verbose = verbose
        self.learning_rate = learning_rate

        self._output_type = "prediction" if net.features[0] == "prediction" \
                            else "sentiment"

        self._success_loss = find_loss("gambling", self._output_type)

        # Ensure there is at least one presenter
        if not presenters:
            raise RuntimeError("At least one data presenter must be given")

    def initialize(self, save: Optional[pathlib.Path] = None,
                   random_init: bool = False) -> None:
        """
        Initializes parameters.  Searches for them in a cache file, or
        initializes them randomly.
        +save+: The path of the saved parameters.  If set, will attempt a load.
                If the load fails, will perform a random_init if random_init
                is set, else it will throw an exception.
        +random_init+: Whether or not to ramdomly initialize parameters.
        """
        if not self.net.trainable:
            return

        if not save or random_init:
            self.net.collect_params().initialize(mx.init.Xavier(),
                                                 ctx=utils.try_gpu(0))
        elif save and (save/self.name).exists():
            self.net.load_parameters(str(save/self.name),
                                     ctx=utils.try_gpu(0))
        else:
            raise RuntimeError(f"Couldn't load file {save/self.name}")

    def save(self, save: pathlib.Path):
        """
        Saves parameters to a file in the +save+ directory.
        """
        if not self.net.trainable:
            return

        save.mkdir(exist_ok=True, parents=True)
        save_file = save/self.name
        self.net.save_parameters(str(save_file))

    def train(self, epochs: int) -> Tuple[List[float], List[float], List[float]]:
        """
        Trains the network for a number of epochs.  Returns the validation
        loss after each epoch, and the percentage-wise success.
        """
        # For each epoch, we generate random data batches.  The DailyDataLoader
        # generates a batch of set size, with samples drawn randomly WITH
        # replacement.  Thus, to simulate a typical epoch, we simply run enough
        # batch sizes that
        #
        #   num_possible_samples = num_batches * size_of_batch
        #
        # We know that we have roughly two years of data to work with, giving
        # ~500 trading days.  Each day, we can count each minute as starting
        # its own sample of size window_size, removing the last
        # (window_size + 10) start minutes because either the window's end or
        # the last 10min_change output will lie outside market hours.  This
        # gives us:
        #
        #   num_samples = 500 * (7.5 * 60 - (window_size + 10))
        #
        # We use a standard batch size of 50 samples, so the total possible
        # batch number comes out to num_samples / 50.  Experience shows that
        # this is brutally too high.  Therefore, we only run 1/10th the
        # possible batches at every epoch.  Finally, note that we reserve one
        # out of every ten batches for validation, trusting the presenter
        # to keep them partitioned from the training set.
        batch_size = 50
        valid_batch_count = (450 - (self.window + 10)) // 10
        training_batch_count = 9 * valid_batch_count

        # First, basic check: is the net even trainable?  If not, just run
        # the validation batch and return it.  Note that we define a special
        # _validate() function here to prevent having to duplicate load for the
        # trainable section.
        def _validate() -> Tuple[float, float]:
            """
            Runs the net on the validation batches.  Returns validation loss
            and success.
            """

            # First, run the model on each validation batch
            val_batches = [self.presenters[0].get_validation_batch(batch_size) \
                           for _ in range(valid_batch_count)]
            valid_loss, valid_success = 0, 0
            print(f"Validating on {valid_batch_count} batches")
            for data_batch, target_batch in val_batches:
                loss, success = self._evaluate(self._format_input(data_batch),
                                               target_batch)
                valid_loss += loss / batch_size
                valid_success += success / batch_size

            # Calculate total loss, accounting for batch size
            valid_loss /= valid_batch_count
            valid_success /= valid_batch_count
            return valid_loss, valid_success

        if not self.net.trainable:
            v_loss, v_success = _validate()
            return [v_loss], [v_loss], [v_success]

        # Initialize trainer
        trainer = mx.gluon.Trainer(self.net.collect_params(), 'adam',
                                   {'learning_rate': self.learning_rate})

        # For each data epoch: train, validate, and record
        training_losses = []
        valid_losses = []
        valid_successes = []
        for epoch in range(epochs):
            training_loss = 0
            print(f"Starting epoch {epoch}...")
            print(f"Training on {training_batch_count} batches")

            # Get a different validation batch each time to minimize
            # validation anomalies
            val_batches = [self.presenters[0].get_validation_batch(batch_size) \
                           for _ in range(valid_batch_count)]

            # Train on each batch
            for batch in range(training_batch_count):
                data_batch, target_batch = \
                        self.presenters[0].get_training_batch(batch_size)
                loss, success = self._evaluate(self._format_input(data_batch),
                                               target_batch,
                                               trainer=trainer)
                loss /= batch_size
                success /= batch_size
                if self.verbose:
                    print(f"E{epoch} B{batch}: Loss = {loss}, Success = {success}")
                training_loss += loss

            # Validate
            valid_loss, valid_success = _validate()
            print(f"Epoch {epoch} validation loss: {valid_loss}")

            training_losses.append(training_loss/training_batch_count)
            valid_losses.append(valid_loss)
            valid_successes.append(valid_success)

        return training_losses, valid_losses, valid_successes

    def predict(self, time: pd.Timestamp) -> pd.DataFrame:
        """
        Makes a prediction for a certain +time+.
        """
        # First initialize the model.  Note that the hidden item may be None.
        # pylint: disable=assignment-from-none
        self._initialize_net(1)

        # Next, get the data
        for presenter in self.presenters:
            input_array = presenter.data_array(time)
            input_frame = presenter.data_frame(time)
            if len(input_array) and len(input_frame):
                break
        if not len(input_array) or not len(input_frame):
            raise RuntimeError(f"No data available for {time}")
        input_array = input_array.reshape(1, input_array.shape[0], -1)

        # Then create a new DataFrame to hold results
        results = pd.DataFrame(columns=['time', 'open', '10min'])
        results.set_index('time')

        # Run the model and show outputs
        print(f"Running prediction for {time}...")
        output = self._iterate_net(self._format_input(input_array))
        print("Done")

        # Generate result dataframe
        results = pd.DataFrame()
        results['time'] = [time]

        if self._output_type == "sentiment":
            results['up'] = [output.asnumpy()[:, 0]]
            results['down'] = [output.asnumpy()[:, 1]]
            results['side'] = [output.asnumpy()[:, 2]]
            results = results.set_index('time')
            results['open'] = [input_frame.open]

        elif self._output_type == "prediction":
            results['output'] = [output.asnumpy().reshape(-1)[-1]]
            results = results.set_index('time')
            results['open'] = [input_frame.open]
            results['10min'] = results['output'] * results['open'] + results['open']

        return results

    def output_type(self) -> str:
        """
        The output produced by this model.
        """
        return self._output_type

    def _initialize_net(self, batch_size: int):
        """
        Initialize the net, returning any hidden state.
        """
        return self.net.begin_state(func=mx.nd.zeros, batch_size=batch_size,
                                    ctx=utils.try_gpu(0))

    def _iterate_net(self, data):
        """
        Runs the net.
        """
        return self.net(data)

    def _evaluate(self, data: mx.nd.NDArray, target: mx.nd.NDArray,
                  trainer: Optional[mx.gluon.Trainer] = None) -> float:
        """
        Runs a batch of +data+ and returns the total loss based on +targett+
        predictions.

        If +trainer+, updates model parameters as well.
        """
        # Initialize loss and model, ensuring that the initial hidden values
        # don't burden the gradient calculation.
        # pylint: disable=assignment-from-none
        batch_size = data.shape[1]
        hidden = self._initialize_net(batch_size)

        if hidden:
            if isinstance(hidden, (tuple, list)):
                # Somehow, this confuses pylint
                # pylint: disable=not-an-iterable
                for val in hidden:
                    val.detach()
            else:
                hidden.detach()

        # Run the model and evaluate loss
        with mx.autograd.record():
            output = self._iterate_net(data)
            # For recurrent models, only train on the last output
            out = output[-1] if len(output.shape) == 3 else output
            i_loss = self.loss(out, target[:, -1, :])
            if trainer:
                i_loss.backward()

        # Add loss to running total
        loss = mx.nd.sum(i_loss).asscalar()

        # If this is a training step, perform a gradient update
        if trainer:
            trainer.step(batch_size)

        # Calculate success
        success = -self._success_loss(out, target.swapaxes(0, 1)[-1]).sum().asscalar()

        return loss, success

    def _format_input(self, array):
        """
        Formats an input array the way this model likes it.
        """
        # Nothing to do here
        return array


class RecurrentModel(SequentialModel):
    """
    Class implementing functions associated with training and using a
    recursive neural net.
    """

    def _format_input(self, array):
        """
        Formats an input array the way this model likes it.
        """
        # Transpose the data, as it's easier to handle gradients when the
        # time axis comes before batch
        return array.swapaxes(0, 1)
