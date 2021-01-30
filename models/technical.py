"""
Daily Technical model.

This model consists of a series of blocks, each of which generates an output
based on a technical indicator.  These outputs are then combined to vote on
a likely market motion.
"""

from typing import Any, Dict, List, Optional

import mxnet as mx

from .model import Net
from .textcnn import TextCNNNet
from .conv import DailyConvolutionalNet


def _get_top_net(name: str) -> mx.gluon.Block:
    """
    Returns a top network for handling votes.  +name+ identifies the type.
    """
    if name == "dense":
        return mx.gluon.nn.Dense(20, activation='tanh')

    if name == "dense60x20":
        seq = mx.gluon.nn.Sequential()
        seq.add(mx.gluon.nn.Dense(60, activation='tanh'))
        seq.add(mx.gluon.nn.Dense(20, activation='tanh'))
        return seq

    if name == "dense-triple":
        seq = mx.gluon.nn.Sequential()
        seq.add(mx.gluon.nn.Dense(20, activation='tanh'))
        seq.add(mx.gluon.nn.Dense(20, activation='tanh'))
        seq.add(mx.gluon.nn.Dense(20, activation='tanh'))
        return seq

    if name == "triplewide":
        seq = mx.gluon.nn.Sequential()
        seq.add(mx.gluon.nn.Dense(30, activation='tanh'))
        seq.add(mx.gluon.nn.Dense(30, activation='tanh'))
        seq.add(mx.gluon.nn.Dense(30, activation='tanh'))
        return seq

    if name == "midstack":
        seq = mx.gluon.nn.Sequential()
        seq.add(mx.gluon.nn.Dense(60, activation='tanh'))
        seq.add(mx.gluon.nn.Dense(45, activation='tanh'))
        seq.add(mx.gluon.nn.Dense(30, activation='tanh'))
        return seq

    if name == "thinstack":
        seq = mx.gluon.nn.Sequential()
        seq.add(mx.gluon.nn.Dense(40, activation='tanh'))
        seq.add(mx.gluon.nn.Dense(30, activation='tanh'))
        seq.add(mx.gluon.nn.Dense(20, activation='tanh'))
        return seq

    if name == "relutivity":
        seq = mx.gluon.nn.Sequential()
        seq.add(mx.gluon.nn.Dense(60, activation='relu'))
        seq.add(mx.gluon.nn.Dense(20, activation='relu'))
        return seq

    if name == "fatstack":
        seq = mx.gluon.nn.Sequential()
        seq.add(mx.gluon.nn.Dense(160, activation='tanh'))
        seq.add(mx.gluon.nn.Dense(80, activation='tanh'))
        seq.add(mx.gluon.nn.Dense(40, activation='tanh'))
        seq.add(mx.gluon.nn.Dense(20, activation='tanh'))
        return seq

    raise RuntimeError(f"No block for {name}")

class TechnicalNet(Net):
    """
    Implementation of the TechnicalNet block per the mxnet framework.
    """
    # Only one public method is needed
    # pylint: disable=too-few-public-methods

    def __init__(self, output: str, features: List[str], singleton: str = None,
            top: Optional[str] = None, extras: List[str] = [],
            **kwargs: Dict[str, Any]):
        """
        Init function.
        """
        super().__init__(**kwargs)
        self.output = output
        self.features = features
        self.extras = extras
        self.top_name = top
        self.singleton = singleton

        if self.singleton and self.output != "sentiment":
            raise RuntimeError("Singleton TI's can only be used with sentiment"
                               "-style outputs")

        with self.name_scope():

            # First, set up the technical feature blocks
            self.subblocks = []
            if self._has_macd():
                self.subblocks.append(MACDBlock(features))
            if self._has_volume():
                self.subblocks.append(VolumeBlock(features))
            if self._has_mass_index():
                self.subblocks.append(MassIndexBlock(features))
            if self._has_trix15():
                self.subblocks.append(TRIXBlock(features, 15))
            if self._has_vortex():
                self.subblocks.append(VortexBlock(features))
            if self._has_rsi():
                self.subblocks.append(RSIBlock(features))
            if self._has_stochastic():
                self.subblocks.append(StochasticOscillatorBlock(features))
            if self._has_slab():
                self.subblocks.append(ExtraBlock("slab", features))
            if self._has_conv():
                self.subblocks.append(ExtraBlock("conv", features))
            if self._has_textcnn():
                self.subblocks.append(ExtraBlock("textcnn", features))
            if self._has_target():
                self.subblocks.append(TargetBlock(features))

            for block in self.subblocks:
                self.register_child(block)

            # Next, set up the vote system
            if not self.singleton:
                if self.top_name:
                    self.top_layer = _get_top_net(self.top_name)
                if self.output == "sentiment":
                    self.collector = mx.gluon.nn.Dense(3)
                else:
                    self.collector = mx.gluon.nn.Dense(1)

    def _has_macd(self) -> bool:
        """
        Are MACD features present
        """
        if self.singleton and self.singleton != 'macd':
            return False
        return 'macd' in self.features and 'macd_signal' in self.features

    def _has_volume(self) -> bool:
        """
        Are volume features present
        """
        if self.singleton and self.singleton != 'volume':
            return False
        return 'volume' in self.features and 'change' in self.features

    def _has_mass_index(self) -> bool:
        """
        Are mass index features present
        """
        if self.singleton and self.singleton != 'mass_index':
            return False
        return 'mass_index' in self.features and 'change' in self.features

    def _has_trix15(self) -> bool:
        """
        Are 15 minute TRIX features present
        """
        if self.singleton and self.singleton != 'trix15':
            return False
        return 'trix15' in self.features

    def _has_vortex(self) -> bool:
        """
        Are Vortex Indicator features present
        """
        if self.singleton and self.singleton != 'vortex':
            return False
        return 'vortex+' in self.features and 'vortex-' in self.features

    def _has_rsi(self) -> bool:
        """
        Are RSI features present
        """
        if self.singleton and self.singleton != 'rsi':
            return False
        return 'rsi' in self.features

    def _has_stochastic(self) -> bool:
        """
        Are RSI features present
        """
        if self.singleton and self.singleton != 'stochastic':
            return False
        return '%D' in self.features and '%K' in self.features

    def _has_slab(self) -> bool:
        """
        Is the Slab multi-layer dense subblock requested
        """
        if self.singleton and self.singleton != 'slab':
            return False
        return 'slab' in self.extras

    def _has_conv(self) -> bool:
        """
        Is a convolutional subblock requested
        """
        if self.singleton and self.singleton != 'conv':
            return False
        return 'conv' in self.extras

    def _has_textcnn(self) -> bool:
        """
        Is a TextCNN subblock requested
        """
        if self.singleton and self.singleton != 'textcnn':
            return False
        return 'textcnn' in self.extras

    def _has_target(self) -> bool:
        """
        Is a Target subblock requested
        """
        if self.singleton and self.singleton != 'target':
            return False
        return 'target' in self.features

    def forward(self, inputs):
        """
        Returns the outputs of the net.
        """
        # First, gather the votes of each technical indicator
        votes = [subblock(inputs) for subblock in self.subblocks]
        if self.singleton:
            # Simply sum and softmax all votes
            output = mx.nd.concat(*[v.reshape(*v.shape, 1) for v in votes],
                                  dim=2)
            output = mx.nd.sum(output, axis=2)
            return mx.ndarray.softmax(output, axis=1)
        output = mx.nd.concat(*votes, dim=1)

        # Next, collect votes and return the prediction
        if self.top_name:
            output = self.top_layer(output)
        output = self.collector(output)
        if self.output == "sentiment":
            output = mx.ndarray.softmax(output, axis=1)
        return output

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
        if not self.singleton or self.singleton in ['conv', 'slab', 'textcnn']:
            return True
        return False

    @property
    def name(self) -> str:
        """
        Returns a name for this net for use in storing parameters and metadata.
        """
        name = "technical"
        if self.top_name:
            name += f"-{self.top_name}"
        if self._has_slab():
            name += "-slab"
        if self._has_conv():
            name += "-conv"
        if self._has_textcnn():
            name += "-textcnn"
        if self._has_volume():
            name += "-volume"
        if self._has_macd():
            name += "-macd"
        if self._has_mass_index():
            name += "-mass_index"
        if self._has_trix15():
            name += "-trix15"
        if self._has_vortex():
            name += "-vortex"
        if self._has_rsi():
            name += "-rsi"
        if self._has_stochastic():
            name += "-stochastic"
        if self._has_target():
            name += "-target"
        return f"{name}-{self.output}"


class MACDBlock(mx.gluon.Block):
    """
    Implementation of the MACD block per the mxnet framework.

    Note that this block only outputs sentiment-style predictions.
    """
    # Only one public method is needed
    # pylint: disable=too-few-public-methods

    def __init__(self, features: List[str], threshold: float = 0.001,
                 **kwargs: Dict[str, Any]):
        """
        Init function.
        """
        super().__init__(**kwargs)

        # Determine which data slices contain MACD and its signal
        if not 'macd' in features or not 'macd_signal' in features:
            raise RuntimeError("Block requires both MACD and MACD Signal features")
        self.macd_index = features.index('macd')
        self.signal_index = features.index('macd_signal')
        self.threshold = threshold

    def forward(self, inputs):
        """
        Returns the outputs of the net.
        """

        # First, calculate the divergence
        macd = inputs[:, :, self.macd_index]
        signal = inputs[:, :, self.signal_index]
        div = (macd[:, -1] - signal[:, -1]) - (macd[:, -2] - signal[:, -2])
        div = div.reshape(-1, 1)
        direction = (macd[:, -1] > signal[:, -1]).reshape(-1, 1)

        # Next, generate sentiments for each divergence
        zeros = mx.nd.zeros(div.shape, ctx=inputs.context)
        ones = mx.nd.ones(div.shape, ctx=inputs.context)

        up_sent = mx.nd.where((div > self.threshold) * direction, ones, zeros)
        down_sent = mx.nd.where((div < -self.threshold) * (1 - direction), ones, zeros)
        side_sent = mx.nd.where(up_sent == down_sent, ones, zeros)
        return mx.nd.concat(up_sent, down_sent, side_sent, dim=1)


class VolumeBlock(mx.gluon.Block):
    """
    Implementation of the volume indicator block per the mxnet framework.

    Note that this block only outputs sentiment-style predictions.
    """
    # Only one public method is needed
    # pylint: disable=too-few-public-methods

    def __init__(self, features: List[str], threshold: float = 1.8,
                 **kwargs: Dict[str, Any]):
        """
        Init function.
        """
        super().__init__(**kwargs)

        # Determine which data slices contain the volume and change
        if not 'volume' in features or not 'change' in features:
            raise RuntimeError("Block requires the volume and change features")
        self.volume_index = features.index('volume')
        self.change_index = features.index('change')
        self.threshold = threshold

    def forward(self, inputs):
        """
        Returns the outputs of the net.
        """

        # First, calculate the average volume
        volume_avg = inputs[:, :, self.volume_index].mean(axis=1).reshape(-1, 1)

        # Next, if the current volume is larger than average * threshold,
        # mark an indicator
        volume_cur = inputs[:, -1, self.volume_index].reshape(-1, 1)
        change_cur = inputs[:, -1, self.change_index].reshape(-1, 1)
        direction = change_cur > 0
        strength = volume_cur > (volume_avg * self.threshold)

        # Next, generate sentiments for each volume movement
        zeros = mx.nd.zeros(strength.shape, ctx=inputs.context)
        ones = mx.nd.ones(strength.shape, ctx=inputs.context)

        up_sent = mx.nd.where(strength * direction, ones, zeros)
        down_sent = mx.nd.where(strength * (1 - direction), ones, zeros)
        side_sent = mx.nd.where(up_sent == down_sent, ones, zeros)
        return mx.nd.concat(up_sent, down_sent, side_sent, dim=1)


class MassIndexBlock(mx.gluon.Block):
    """
    Implementation of the mass index block per the mxnet framework.

    Note that this block only outputs sentiment-style predictions.
    """
    # Only one public method is needed
    # pylint: disable=too-few-public-methods

    def __init__(self, features: List[str], **kwargs: Dict[str, Any]):
        """
        Init function.
        """
        super().__init__(**kwargs)

        # Determine which data slices contain the Mass Index and Change
        if not 'mass_index' in features or not 'change' in features:
            raise RuntimeError("Block requires the Mass Index and Change features")
        self.mass_index = features.index('mass_index')
        self.change_index = features.index('change')

    def forward(self, inputs):
        """
        Returns the outputs of the net.
        """
        # First, determine if the "reversal bulge" has been found.
        bulge = mx.nd.sum(inputs[:, -10:-1, self.mass_index] > (26.0/250.0),
                          axis=1) > 0
        drop = inputs[:, -1, self.mass_index] < (25.75/250.0)
        reversal = (bulge * drop).reshape(-1, 1)

        # Next, determine trend direction by summing changes over the last
        # ten periods
        change = mx.nd.sum(inputs[:, -10:-1, self.change_index],
                           axis=1).reshape(-1, 1)
        uptrend = change > 0

        # Next, generate sentiments
        zeros = mx.nd.zeros(reversal.shape, ctx=inputs.context)
        ones = mx.nd.ones(reversal.shape, ctx=inputs.context)

        up_sent = mx.nd.where((1 - uptrend) * reversal, ones, zeros)
        down_sent = mx.nd.where(uptrend * reversal, ones, zeros)
        side_sent = mx.nd.where(up_sent == down_sent, ones, zeros)
        return mx.nd.concat(up_sent, down_sent, side_sent, dim=1)


class TRIXBlock(mx.gluon.Block):
    """
    Implementation of the TRIX indicator block per the mxnet framework.

    Note that this block only outputs sentiment-style predictions.
    """
    # Only one public method is needed
    # pylint: disable=too-few-public-methods

    def __init__(self, features: List[str], period: int,
                 threshold: float = 0.0001, **kwargs: Dict[str, Any]):
        """
        Init function.
        """
        super().__init__(**kwargs)

        # Determine which data slices contain the TRIX indicator
        if not f"trix{period}" in features:
            raise RuntimeError(f"Block requires the trix{period} feature")
        self.trix_index = features.index(f"trix{period}")
        self.threshold = threshold

    def forward(self, inputs):
        """
        Returns the outputs of the net.
        """

        # First, extract the TRIX indicator
        trix = inputs[:, -1, self.trix_index].reshape(-1, 1)

        # Next, generate sentiments, indicating a movement when the threshold
        # is exceeded.
        zeros = mx.nd.zeros(trix.shape, ctx=inputs.context)
        ones = mx.nd.ones(trix.shape, ctx=inputs.context)

        up_sent = mx.nd.where(trix > self.threshold, ones, zeros)
        down_sent = mx.nd.where(trix < -self.threshold, ones, zeros)
        side_sent = mx.nd.where(up_sent == down_sent, ones, zeros)
        return mx.nd.concat(up_sent, down_sent, side_sent, dim=1)


class VortexBlock(mx.gluon.Block):
    """
    Implementation of the Vortex Indicator block per the mxnet framework.

    Note that this block only outputs sentiment-style predictions.
    """
    # Only one public method is needed
    # pylint: disable=too-few-public-methods

    def __init__(self, features: List[str], **kwargs: Dict[str, Any]):
        """
        Init function.
        """
        super().__init__(**kwargs)

        # Determine which data slices contain the V+ and V- signals
        if not 'vortex+' in features or not 'vortex-' in features:
            raise RuntimeError("Block requires both Vortex + and - features")
        self.vortex_p_index = features.index('vortex+')
        self.vortex_n_index = features.index('vortex-')

    def forward(self, inputs):
        """
        Returns the outputs of the net.
        """
        # Find where the vortex indicators cross
        vp1 = inputs[:, -1, self.vortex_p_index]
        vn1 = inputs[:, -1, self.vortex_n_index]
        vp2 = inputs[:, -2, self.vortex_p_index]
        vn2 = inputs[:, -2, self.vortex_n_index]
        cross_up = ((vp2 <= vn2) * (vp1 > vn1)).reshape(-1, 1)
        cross_down = ((vp2 >= vn2) * (vp1 < vn1)).reshape(-1, 1)

        # Next, generate and return sentiments for each cross
        zeros = mx.nd.zeros(cross_up.shape, ctx=inputs.context)
        ones = mx.nd.ones(cross_up.shape, ctx=inputs.context)
        side_sent = mx.nd.where(cross_up == cross_down, ones, zeros)
        return mx.nd.concat(cross_up, cross_down, side_sent, dim=1)


class RSIBlock(mx.gluon.Block):
    """
    Implementation of the RSI indicator block per the mxnet framework.

    Note that this block only outputs sentiment-style predictions.
    """
    # Only one public method is needed
    # pylint: disable=too-few-public-methods

    def __init__(self, features: List[str], threshold: float = 1.8,
                 **kwargs: Dict[str, Any]):
        """
        Init function.
        """
        super().__init__(**kwargs)

        # Determine which data slice contains the RSI
        if not 'rsi' in features:
            raise RuntimeError("Block requires the RSI feature")
        self.rsi_index = features.index('rsi')
        self.threshold = threshold

    def forward(self, inputs):
        """
        Returns the outputs of the net.
        """
        # First, extract the RSI and create corresponding constant arrays
        rsi = inputs[:, -1, self.rsi_index].reshape(-1, 1)
        zeros = mx.nd.zeros(rsi.shape, ctx=inputs.context)
        ones = mx.nd.ones(rsi.shape, ctx=inputs.context)

        # For each level or RSI, make a prediction, increasing likelihood as
        # the RSI drifts further into overbought/oversold territory
        up_70 = mx.nd.where(rsi > 0.7, ones/4, zeros)
        up_75 = mx.nd.where(rsi > 0.75, ones/4, zeros)
        up_80 = mx.nd.where(rsi > 0.8, ones/2, zeros)
        down_30 = mx.nd.where(rsi < 0.3, ones/4, zeros)
        down_25 = mx.nd.where(rsi < 0.25, ones/4, zeros)
        down_20 = mx.nd.where(rsi < 0.2, ones/2, zeros)
        up_sent = up_70 + up_75 + up_80
        down_sent = down_30 + down_25 + down_20
        side_sent = 1 - (up_sent + down_sent)

        # Return the predictions
        return mx.nd.concat(up_sent, down_sent, side_sent, dim=1)


class StochasticOscillatorBlock(mx.gluon.Block):
    """
    Implementation of the StochasticOscillator block per the mxnet framework.

    Note that this block only outputs sentiment-style predictions.
    """
    # Only one public method is needed
    # pylint: disable=too-few-public-methods

    def __init__(self, features: List[str], **kwargs: Dict[str, Any]):
        """
        Init function.
        """
        super().__init__(**kwargs)

        # Determine which data slices contain %K and %D
        if not '%K' in features or not '%D' in features:
            raise RuntimeError("Block requires both %K and %D features")
        self.pK_index = features.index('%K')
        self.pD_index = features.index('%D')

    def forward(self, inputs):
        """
        Returns the outputs of the net.
        """

        # Find where the indicators cross in extreme territory
        pK = inputs[:, :, self.pK_index]
        pD = inputs[:, :, self.pD_index]
        extreme_up, extreme_down = pK[:, -1] > .8, pK[:, -1] < .2
        cur_diff, last_diff = (pK[:, -1] - pD[:, -1]), (pK[:, -2] - pD[:, -2])
        cross_over = (last_diff < 0) * (cur_diff >= 0)
        cross_under = (last_diff > 0) * (cur_diff <= 0)

        # Generate and return sentiments
        up_sent = (cross_over * extreme_down).reshape(-1, 1)
        down_sent = (cross_under * extreme_up).reshape(-1, 1)
        side_sent = 1 - (up_sent + down_sent)
        return mx.nd.concat(up_sent, down_sent, side_sent, dim=1)


class ExtraBlock(mx.gluon.Block):
    """
    Block for including other nets.  Responsible for extracting useful
    features from the master input and feeding them into the wrapped net.
    Note that this block only outputs sentiment-style predictions.
    """
    # Only one public method is needed
    # pylint: disable=too-few-public-methods

    def __init__(self, net: str, features: List[str], **kwargs: Dict[str, Any]):
        """
        Init function.
        """
        super().__init__(**kwargs)

        # Gather features
        self.features = []
        for feature in ["high", "low", "change", "open", "volume", "time"]:
            if feature in features:
                self.features.append(features.index(feature))

        with self.name_scope():
            if net == "conv":
                self.net = DailyConvolutionalNet("sentiment", self.features)
            elif net == "textcnn":
                self.net = TextCNNNet("sentiment", self.features)
            elif net == "slab":
                self.net = mx.gluon.nn.Sequential()
                self.net.add(mx.gluon.nn.Dense(180, activation='tanh'))
                self.net.add(mx.gluon.nn.Dense(60, activation='tanh'))
                self.net.add(mx.gluon.nn.Dense(20, activation='tanh'))
                self.net.add(mx.gluon.nn.Dense(3, activation='tanh'))
            else:
                raise RuntimeError(f"Couldn't identify net '{net}'")

    def forward(self, inputs):
        """
        Returns the outputs of the net.
        """
        # Only include the standard features
        standard_inputs = inputs[:, :, self.features]

        # Run the model
        return self.net(standard_inputs)


class TargetBlock(mx.gluon.Block):
    """
    Block for comparing predictions against the theoretical maximum.  Assumes
    that a magical oracle provides 100% accurate predictions, and simply
    selects market sentiment using them.

    Note that this block only outputs sentiment-style predictions.
    """
    # Only one public method is needed
    # pylint: disable=too-few-public-methods

    def __init__(self, features: List[str], threshold: float = 1.8,
                 **kwargs: Dict[str, Any]):
        """
        Init function.
        """
        super().__init__(**kwargs)

        # Determine which data slice contains the RSI
        if not 'target' in features:
            raise RuntimeError("Block requires the Target feature")
        self.target_index = features.index('target')

    def forward(self, inputs):
        """
        Returns the outputs of the net.
        """
        # Extract the prediction and return the sentiment
        target = inputs[:, -1, self.target_index].reshape(-1, 1)
        zeros = mx.nd.zeros(target.shape, ctx=inputs.context)
        return mx.nd.concat(target >= 0, target < 0, zeros, dim=1)
