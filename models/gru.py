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

    def __init__(self, features: List[List[str]], output: str = "sentiment",
                 num_hidden: int = 64, num_layers: int = 1, dropout: int = 0.1,
                 **kwargs: Dict[str, Any]) -> None:
        """
        Init function.

        +output+: The type of output tensor
        +num_hidden+: Number of nodes in hidden layer (default 256)
        +num_layers+: Number of recurrent layers (default 1)
        +dropout+: Training dropout coefficient
        """
        super().__init__(features, **kwargs)
        self.output = output
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

    def begin_state(self, *args, **kwargs) -> List[Any]:
        """
        Get beginning state.
        """
        return self.rnn.begin_state(*args, **kwargs)

    @property
    def features(self) -> List[str]:
        """
        A list of features output by this net.
        """
        if self.output == "sentiment":
            return ["up", "down", "side"]
        else:
            return ["prediction"]

    @property
    def trainable(self) -> bool:
        """
        Whether or not this net is trainable
        """
        return True
