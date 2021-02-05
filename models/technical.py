"""
Daily Technical model.

This model consists of a series of blocks, each of which generates an output
based on a technical indicator.  These outputs are then combined to vote on
a likely market motion.
"""

from typing import Any, Dict, List, Optional

import mxnet as mx

from .model import Net


class TechnicalBlock(Net):
    """
    The base technical indicator block.
    """
    # Only one public method is needed
    # pylint: disable=too-few-public-methods

    @property
    def features(self) -> List[str]:
        """
        A list of features output by this net.  Note that all technical
        indicator blocks are sentiment-style.
        """
        return ["up", "down", "side"]

    @property
    def trainable(self) -> bool:
        """
        Whether or not this net is trainable.  Technical indicators are not.
        """
        return False


class MACDBlock(TechnicalBlock):
    """
    Implementation of the MACD block per the mxnet framework.
    """
    # Only one public method is needed
    # pylint: disable=too-few-public-methods

    def __init__(self, features: List[List[str]], threshold: float = 0.001,
                 **kwargs: Dict[str, Any]):
        """
        Init function.
        """
        super().__init__(features, **kwargs)

        # Determine which data slices contain MACD and its signal
        if not 'macd' in features[0] or not 'macd_signal' in features[0]:
            raise RuntimeError("Block requires both MACD and MACD Signal features")
        self.macd_index = features[0].index('macd')
        self.signal_index = features[0].index('macd_signal')
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


class VolumeBlock(TechnicalBlock):
    """
    Implementation of the volume indicator block per the mxnet framework.
    """
    # Only one public method is needed
    # pylint: disable=too-few-public-methods

    def __init__(self, features: List[List[str]], threshold: float = 1.8,
                 **kwargs: Dict[str, Any]):
        """
        Init function.
        """
        super().__init__(features, **kwargs)

        # Determine which data slices contain the volume and change
        if not 'volume' in features[0] or not 'change' in features[0]:
            raise RuntimeError("Block requires the volume and change features")
        self.volume_index = features[0].index('volume')
        self.change_index = features[0].index('change')
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


class MomentumBlock(TechnicalBlock):
    """
    Implementation of the Momentum block per the mxnet framework.
    """
    # Only one public method is needed
    # pylint: disable=too-few-public-methods

    def __init__(self, features: List[List[str]], threshold: float = 0.001,
                 **kwargs: Dict[str, Any]):
        """
        Init function.
        """
        super().__init__(features, **kwargs)

        # Determine which data slices contain %K and %D
        if not 'open' in features[0]:
            raise RuntimeError("Block requires Open features")
        self.open_index = features[0].index('open')
        self.threshold = threshold

    def forward(self, inputs):
        """
        Returns the outputs of the net.
        """
        # Find out which way the stock is moving
        start = inputs[:, -40, self.open_index]
        mid = inputs[:, -20, self.open_index]
        end = inputs[:, -1, self.open_index]

        # If the motion is consistent, assume it will continue.  Else, don't
        # assume.
        t = self.threshold
        up_sent = ((start < (mid - t)) * (mid < (end - t))).reshape(-1, 1)
        down_sent = ((start > (mid + t)) * (mid > (end + t))).reshape(-1, 1)
        side_sent = 1 - (up_sent + down_sent)
        return mx.nd.concat(up_sent, down_sent, side_sent, dim=1)


class MassIndexBlock(TechnicalBlock):
    """
    Implementation of the mass index block per the mxnet framework.
    """
    # Only one public method is needed
    # pylint: disable=too-few-public-methods

    def __init__(self, features: List[List[str]], **kwargs: Dict[str, Any]):
        """
        Init function.
        """
        super().__init__(features, **kwargs)

        # Determine which data slices contain the Mass Index and Change
        if not 'mass_index' in features[0] or not 'change' in features[0]:
            raise RuntimeError("Block requires the Mass Index and Change features")
        self.mass_index = features[0].index('mass_index')
        self.change_index = features[0].index('change')

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


class TRIXBlock(TechnicalBlock):
    """
    Implementation of the TRIX indicator block per the mxnet framework.
    """
    # Only one public method is needed
    # pylint: disable=too-few-public-methods

    def __init__(self, features: List[List[str]], period: int = 15,
                 threshold: float = 0.0001, **kwargs: Dict[str, Any]):
        """
        Init function.
        """
        super().__init__(features, **kwargs)

        # Determine which data slices contain the TRIX indicator
        if not f"trix{period}" in features[0]:
            raise RuntimeError(f"Block requires the trix{period} feature")
        self.trix_index = features[0].index(f"trix{period}")
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


class VortexBlock(TechnicalBlock):
    """
    Implementation of the Vortex Indicator block per the mxnet framework.
    """
    # Only one public method is needed
    # pylint: disable=too-few-public-methods

    def __init__(self, features: List[List[str]], **kwargs: Dict[str, Any]):
        """
        Init function.
        """
        super().__init__(features, **kwargs)

        # Determine which data slices contain the V+ and V- signals
        if not 'vortex+' in features[0] or not 'vortex-' in features[0]:
            raise RuntimeError("Block requires both Vortex + and - features")
        self.vortex_p_index = features[0].index('vortex+')
        self.vortex_n_index = features[0].index('vortex-')

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


class RSIBlock(TechnicalBlock):
    """
    Implementation of the RSI indicator block per the mxnet framework.
    """
    # Only one public method is needed
    # pylint: disable=too-few-public-methods

    def __init__(self, features: List[List[str]], threshold: float = 1.8,
                 **kwargs: Dict[str, Any]):
        """
        Init function.
        """
        super().__init__(features, **kwargs)

        # Determine which data slice contains the RSI
        if not 'rsi' in features[0]:
            raise RuntimeError("Block requires the RSI feature")
        self.rsi_index = features[0].index('rsi')
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


class StochasticOscillatorBlock(TechnicalBlock):
    """
    Implementation of the StochasticOscillator block per the mxnet framework.
    """
    # Only one public method is needed
    # pylint: disable=too-few-public-methods

    def __init__(self, features: List[List[str]], **kwargs: Dict[str, Any]):
        """
        Init function.
        """
        super().__init__(features, **kwargs)

        # Determine which data slices contain %K and %D
        if not '%K' in features[0] or not '%D' in features[0]:
            raise RuntimeError("Block requires both %K and %D features")
        self.pK_index = features[0].index('%K')
        self.pD_index = features[0].index('%D')

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


class WilliamsBlock(TechnicalBlock):
    """
    Implementation of the Willams %R indicator block per the mxnet framework.
    """
    # Only one public method is needed
    # pylint: disable=too-few-public-methods

    def __init__(self, features: List[List[str]], threshold: float = 1.8,
                 **kwargs: Dict[str, Any]):
        """
        Init function.
        """
        super().__init__(features, **kwargs)

        if not '%R' in features[0]:
            raise RuntimeError("Block requires the %R feature")
        self.pR_index = features[0].index('%R')

    def forward(self, inputs):
        """
        Returns the outputs of the net.
        """
        # First, extract the %R
        pR = inputs[:, :, self.pR_index]

        # Next, look for the overbought/oversold indicators
        up_sent = ((pR[:, -6]  == -1.0) * (pR[:, -1] > -.95)).reshape(-1, 1)
        down_sent = ((pR[:, -6] == 0) * (pR[:, -1] < -.05)).reshape(-1, 1)
        side_sent = 1 - (up_sent + down_sent)

        # Return the predictions
        return mx.nd.concat(up_sent, down_sent, side_sent, dim=1)


class TargetBlock(TechnicalBlock):
    """
    Block for comparing predictions against the theoretical maximum.  Assumes
    that a magical oracle provides 100% accurate predictions, and simply
    selects market sentiment using them.
    """
    # Only one public method is needed
    # pylint: disable=too-few-public-methods

    def __init__(self, features: List[str], threshold: float = 1.8,
                 **kwargs: Dict[str, Any]):
        """
        Init function.
        """
        super().__init__(features, **kwargs)

        # Determine which data slice contains the RSI
        if not 'target' in features[0]:
            raise RuntimeError("Block requires the Target feature")
        self.target_index = features[0].index('target')

    def forward(self, inputs):
        """
        Returns the outputs of the net.
        """
        # Extract the prediction and return the sentiment
        target = inputs[:, -1, self.target_index].reshape(-1, 1)
        zeros = mx.nd.zeros(target.shape, ctx=inputs.context)
        return mx.nd.concat(target >= 0, target < 0, zeros, dim=1)
