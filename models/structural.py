"""
Structural Nets

This module consists of a set of nets designed for stitching other nets
together within a model.
"""

from typing import List

import mxnet as mx

from . import Net

class ConcatenateNet(Net):
    """
    Net for concatenating arrays
    """
    # Only one public method is needed
    # pylint: disable=too-few-public-methods

    def __init__(self, data_features: List[str], axis: int = 1, **kwargs) \
            -> None:
        """
        Init function.  Takes the typical feature list and and +axis+ along
        which to concatenate.
        """
        super().__init__(data_features, **kwargs)
        self._axis = axis

    def forward(self, *inputs):
        """
        Forward function.  Concatenates +inputs+.
        """
        return mx.nd.concat(*inputs, dim=self._axis)

    @property
    def features(self) -> List[str]:
        """
        A list of features output by this net.
        """
        # In theory, features shouldn't change
        return self._input_features

    @property
    def trainable(self) -> bool:
        """
        Whether or not this net is trainable
        """
        return False
