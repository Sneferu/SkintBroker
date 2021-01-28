"""
Daily MultiConv model.

This model is based off the TextCNN architecture used for textual sentiment
analysis, but extended to use a more complex feature space.  It is predicated
on the idea that many concepts from NLP are also applicable to market analysis,
most notably contextual embedding and variable size token groupings.

It consists of three layers.  The first is a simple multi-layer dense net run
on each data frame.  This serves the same purpose as an embedding layer in NLP,
though the outputs are larger than the inputs and highly nonlinear.  It allows
various per-minute movements to be mapped into a more complex feature space.

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

from typing import Any, Dict, Tuple

import mxnet as mx

class MultiConvComponent(mx.gluon.Block):
    """
    Implementation of the MultiConvComponent block per the mxnet framework.
    """
    # Only one public method is needed
    # pylint: disable=too-few-public-methods

    def __init__(self, output: str, windows: Tuple[int], **kwargs: Dict[str, Any]):
        """
        Init function.
        """
        super().__init__(**kwargs)
        self.output = output
        with self.name_scope():

            # First, set up the embedding blocks
            self.encoder1 = mx.gluon.nn.Dense(40, flatten=False, activation='tanh')
            self.encoder2 = mx.gluon.nn.Dense(40, flatten=False, activation='tanh')

            # Next, set up convolutional layers with windows of the requested
            # size.
            convs = []
            for window in windows:
                conv = mx.gluon.nn.Conv1D(channels=40, kernel_size=window,
                                          strides=1, activation='tanh')
                convs.append(conv)

            # Set up the decoder layer with 20 possible movements and one
            # prediction
            self.decoder1 = mx.gluon.nn.Dense(29)
            if self.output == "sentiment":
                self.decoder2 = mx.gluon.nn.Dense(4)
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
        convs = [conv(embedding).flatten() for conv in self.convs]
        votes = mx.nd.concat(*convs, dim=1)

        # Finally, decode votes and return the prediction
        output = self.decoder2(self.decoder1(votes))
        if self.output == "sentiment":
            sentiment = mx.ndarray.softmax(output[:, :2], axis=1)
            confidence = mx.ndarray.L2Normalization(output[:, 2:])[:, 0]
            output = mx.nd.concat(sentiment, confidence.reshape(-1, 1), dim=1)
        return output


class DailyCompressorBlock(mx.gluon.Block):
    """
    Implementation of the DailyCompressorBlock per the mxnet framework.

    This block is responsible for taking daily stock data with one sample
    frequency and compressing it to a larger sample frequency.  As such,
    it requires a) an intimate understanding of the daily data format and b)
    inputs which are exactly divisible by the compression ratio.
    """
    # Only one public method is needed
    # pylint: disable=too-few-public-methods

    def __init__(self, compression: int, **kwargs: Dict[str, Any]):
        """
        Init function.
        """
        super().__init__(**kwargs)
        self.compression = compression

    def forward(self, inputs):
        """
        Returns the outputs of the block.
        """

        # Start with daily change
        daily_change = inputs[:, (window-1)::window, 0]

        # Next, local change

        # TODO
        

        return output


class MultiConvNet(mx.gluon.Block):
    """
    Implementation of the MultiConv net per the mxnet framework.
    """
    # Only one public method is needed
    # pylint: disable=too-few-public-methods

    def __init__(self, output: str, **kwargs: Dict[str, Any]):
        """
        Init function.
        """
        super().__init__(**kwargs)
        self.output = output
        with self.name_scope():


            # First, set up the necessary compression blocks
            self.comp3 = DailyCompressorBlock(3)
            self.comp5 = DailyCompressorBlock(5)
            self.comp15 = DailyCompressorBlock(15)

            # Next, create a MultiConv component for each compression level
            self.mcs = [MultiConvComponent(output, (2, 3, 5, 10))
                        MultiConvComponent(output, (2, 3, 5))
                        MultiConvComponent(output, (2, 3))
                        MultiConvComponent(output, (2))]

            # Finally, set up the output decoder block
            if self.output == "sentiment":
                self.decoder = mx.gluon.nn.Dense(4)
            else:
                self.decoder = mx.gluon.nn.Dense(1)

    def forward(self, inputs):
        """
        Returns the outputs of the net.
        """

        # First, compress the data
        comps = [inputs, self.comp3(inputs), self.comp5(inputs),
                 self.comp15(inputs)]

        # Next, run each compression frame through the corresponding
        # MultiConv block
        votes = []
        for index, comp in enumerate(comps):
            votes.append(self.mcs[index](comp).flatten())

        # Finally, concatenate votes and run them through the output
        # decoder.
        output = self.decoder(mx.nd.concat(*votes, dim=1))

        # Finally, return the prediction
        output = self.decoder2(self.decoder1(votes))
        if self.output == "sentiment":
            sentiment = mx.ndarray.softmax(output[:, :2], axis=1)
            confidence = mx.ndarray.L2Normalization(output[:, 2:])[:, 0]
            output = mx.nd.concat(sentiment, confidence.reshape(-1, 1), dim=1)
        return output

    @property
    def name(self) -> str:
        """
        Returns a name for this net for use in storing parameters and metadata.
        """
        return f"multiconv-{self.output}"

