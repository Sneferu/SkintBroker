"""
Daily Recurrent Model.

This is the simplest model in SkintBroker.  It is simply an RNN which searches
for patterns in daily market movements and tries to predict them.
"""

from typing import Any, Dict, List

import mxnet as mx

from .model import Net


class DailyRecurrentNet(Net):
    """
    Implementation of the DailyRecurrentNet block per the mxnet framework.
    """

    def __init__(self, output: str, features: List[str], num_hidden: int = 64,
                 num_layers: int = 1, dropout: int = 0.1, **kwargs: Dict[str, Any]):
        """
        Init function.

        +output+: The type of output tensor
        +num_hidden+: Number of nodes in hidden layer (default 256)
        +num_layers+: Number of recurrent layers (default 1)
        +activation+: Activation function (tanh or relu, default relu)
        +dropout+: Training dropout coefficient
        """
        super().__init__(**kwargs)
        self.output = output
        self.features = features
        self.num_hidden = num_hidden
        self.num_layers = num_layers
        with self.name_scope():

            # Set up dropout
            self.drop = mx.gluon.nn.Dropout(dropout)

            # Set up recurrent block
            self.rnn = mx.gluon.rnn.GRU(num_hidden, num_layers,
                                        dropout=dropout)

            # Set up a decoder layer
            self.decoder1 = mx.gluon.nn.Dense(16, flatten=False)

            # Add final prediction output
            if self.output == "sentiment":
                self.prediction = mx.gluon.nn.Dense(3, flatten=False)
            else:
                self.prediction = mx.gluon.nn.Dense(1, flatten=False)


    def forward(self, inputs, hidden):
        """
        Returns the outputs of the net.
        """
        drop = self.drop(inputs)
        output, hidden = self.rnn(drop, hidden)
        output = self.drop(output)
        output = self.decoder1(output)
        output = self.prediction(output)
        if self.output == "sentiment":
            output = mx.ndarray.softmax(output, axis=2)

        return output, hidden

    def begin_state(self, *args, **kwargs):
        """
        Get beginning state.
        """
        return self.rnn.begin_state(*args, **kwargs)

    @property
    def output_format(self) -> str:
        """
        The output format of this net
        """
        return self.output

    @property
    def trainable(self) -> bool:
        """
        Whether or not this net is trainable
        """
        return True

    @property
    def name(self) -> str:
        """
        Returns a name for this net for use in storing parameters and metadata.
        """
        name = f"rnn-{self.num_hidden}-{self.num_layers}"
        for feature in self.features:
            name += f"-{feature}"
        return f"{name}-{self.output}"
