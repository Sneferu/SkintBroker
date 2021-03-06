"""
Daily Convolutional Model.

This is the simplest convolutional model in SkintBroker.  It uses a two-level
convolutional network to identify market patters over a moving window, making
a prediction for ten minutes hence.
"""

from typing import Any, Dict, List

import mxnet as mx

from .model import Net


class DailyConvolutionalNet(Net):
    """
    Implementation of the DailyConvolutionalNet block per the mxnet framework.
    """
    # Only one public method is needed
    # pylint: disable=too-few-public-methods

    def __init__(self, features: List[List[str]], output: str = "sentiment",
                 **kwargs: Dict[str, Any]) -> None:
        """
        Init function.
        """
        super().__init__(features, **kwargs)
        self.output = output

        with self.name_scope():

            # Set up two convolutional layers
            self.conv1 = mx.gluon.nn.Conv1D(channels=20, kernel_size=9,
                                            strides=1, activation='tanh')
            self.pool1 = mx.gluon.nn.MaxPool1D(pool_size=2, strides=2)
            self.conv2 = mx.gluon.nn.Conv1D(channels=60, kernel_size=3,
                                            strides=1, activation='tanh')
            self.pool2 = mx.gluon.nn.MaxPool1D(pool_size=2, strides=2)

            # And a decoder
            self.decoder1 = mx.gluon.nn.Dense(10)

            # Finally, set up a dense prediction layer
            if output == "sentiment":
                self.prediction = mx.gluon.nn.Dense(3)
            else:
                self.prediction = mx.gluon.nn.Dense(1)

    def forward(self, inputs):
        """
        Returns the outputs of the net.
        """
        output = self.conv1(inputs.swapaxes(1,2))
        output = self.pool1(output)
        output = self.conv2(output)
        output = self.pool2(output)
        output = self.decoder1(output)
        output = self.prediction(output)
        if self.output == "sentiment":
            output = mx.ndarray.softmax(output, axis=1)
        return output

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
