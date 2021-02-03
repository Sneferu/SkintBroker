"""
Dense Net

This module contains a class for constructing various topologies of dense
net.
"""

from typing import Any, Dict, List, Optional

import mxnet as mx

from .model import Net


class DenseNet(Net):
    """
    Implementation of the DenseNet block per the mxnet framework.
    """
    # Only one public method is needed
    # pylint: disable=too-few-public-methods

    def __init__(self, features: List[str], activations: List[str] = [],
                 sizes: List[int] = [], output: str = "sentiment",
                 **kwargs: Dict[str, Any]) -> None:
        """
        Init function.
        """
        super().__init__(features, **kwargs)

        self._output = output

        # Check sizes and activations lists
        if len(sizes) != len(activations):
            raise RuntimeError("Same number of sizes and activations needed!")

        with self.name_scope():

            # First, create sequential net
            self.seq = mx.gluon.nn.Sequential()

            # For each size/activation, add a corresponding dense layer
            for index, size in enumerate(sizes):
                self.seq.add(mx.gluon.nn.Dense(size,
                                               activation=activations[index]))
            # Finally, add the last dense layer
            if self._output == "sentiment":
                self.seq.add(mx.gluon.nn.Dense(3, activation='sigmoid'))
            else:
                self.seq.add(mx.gluon.nn.Dense(1))

    @property
    def features(self) -> List[str]:
        """
        A list of features output by this net.
        """
        if self._output == "sentiment":
            return ["up", "down", "side"]
        else:
            return ["prediction"]

    @property
    def trainable(self) -> bool:
        """
        Whether or not this net is trainable
        """
        return True

    def forward(self, inputs):
        """
        Returns the outputs of the net.
        """
        output = self.seq(inputs)
        if self._output == "sentiment":
            output = mx.ndarray.softmax(output, axis=1)
        return output
