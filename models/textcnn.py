"""
Daily TextCNN model.

This model is based off the TextCNN architecture used for textual sentiment
analysis.  It is predicated on the idea that many concepts from NLP are also
applicable to market analysis, most notably contextual embedding and variable
size token groupings.

It consists of three layers.  The first is a simple dense net run on each
data frame.  This serves the same purpose as an embedding layer in NLP,
though the "dictionary" is more compact and continuous.  It allows various
per-minute movements to be mapped into a more complex vector space.

The second consists of multiple convolution layers, each with a different
window size.  This enables detection of features with different lengths, and
all results are concatenated into a master feature vector.  Note that unlike
the standard TextCNN model, there is no pooling layer here.  This is because
the decoder vector has every right to assign weights based on a feature's
place in the sequence, so that data must be preserved.

Finally, a decoding layer consists of three dense layers, consisting of the
one present in the original TextCNN model which outputs a classification
vector, and one which converts this vector into an actual percentage change.
"""

from typing import Any, Dict, List

import mxnet as mx

from .model import Net

class TextCNNNet(Net):
    """
    Implementation of the TextCNN block per the mxnet framework.
    """
    # Only one public method is needed
    # pylint: disable=too-few-public-methods

    def __init__(self, features: List[List[str]], output: str = "sentiment",
                 **kwargs: Dict[str, Any]) -> None:
        """
        Init function.
        """
        self.output = output
        super().__init__(features, **kwargs)
        with self.name_scope():

            # First, set up the embedding blocks
            self.encoder1 = mx.gluon.nn.Dense(20, flatten=False, activation='tanh')
            self.encoder2 = mx.gluon.nn.Dense(20, flatten=False, activation='tanh')

            # Next, set up convolutional layers with windows of size 2, 3, 5,
            # and 10
            self.conv2 = mx.gluon.nn.Conv1D(channels=30, kernel_size=2,
                                            strides=1, activation='relu')
            self.conv3 = mx.gluon.nn.Conv1D(channels=30, kernel_size=3,
                                            strides=1, activation='relu')
            self.conv5 = mx.gluon.nn.Conv1D(channels=30, kernel_size=5,
                                            strides=1, activation='relu')
            self.conv10 = mx.gluon.nn.Conv1D(channels=30, kernel_size=10,
                                             strides=1, activation='relu')

            # Set up the decoder layer with 20 possible movements and one
            # prediction
            self.decoder1 = mx.gluon.nn.Dense(29)
            if self.output == "sentiment":
                self.decoder2 = mx.gluon.nn.Dense(3)
            else:
                self.decoder2 = mx.gluon.nn.Dense(1)

    def forward(self, inputs):
        """
        Returns the outputs of the net.
        """
        # First, embed into the higher space
        embedding = self.encoder2(self.encoder1(inputs)).swapaxes(1, 2)

        # Next, run each convolution layer on the embedded data and flatten
        # into a vote vector
        convs = [conv(embedding).flatten() for conv in \
                 [self.conv2, self.conv3, self.conv5, self.conv10]]
        votes = mx.nd.concat(*convs, dim=1)

        # Finally, decode votes and return the prediction
        output = self.decoder2(self.decoder1(votes))
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
