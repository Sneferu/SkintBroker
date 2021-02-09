"""
Loss functions.

This module consists of a set of Loss blocks for use with SkintBroker models,
as well as the find_loss() function designed to instantiate an appropriate
one.  Many of these loss functions are simply wrappers for mxnet built-in
types.  Beyond that, there is the Gambling Loss and its gradient-friendly
cousin, the GFG loss, described below.

Beyond the purchase of simple equities, numerous financial derivatives allow
one to bet on market motions in either direction, with gain or loss tied both
to the direction and size of the change.  For instance, call options will
increase in value any time the underlying equity rises, with performance
proportional to the rise, where a put will perform the opposite.

In order to make money betting on such items, it is important that the total
magnitude of changes moving in the predicted direction is greater than the
total magnitude of the changes moving in the opposite direction, summed over
a number of bets.  This is the metric used by the Gambler's Loss.

The prediction tensor consists of one or more values, i.e the predicted
percent change associated with an underlying asset.  The target tensor consists
of the actual change.  If the magnitude of the target is lower than a preset
value, no loss is assigned, as no trades are assumed on tiny motions.  For
larger predictions, a loss equivalent to the magnitude of the motion of the
underlying asset is subtracted for correct predictions, and added for
incorrect ones, based on the idea that bets are made not on the predicted
magnitude, but only on the predicted directions.

Over training, the prediction will slowly approximate the degree of certainty
associated with a move, rather than the size of the move itself.  Assuming
any linearity in the markets, these will be roughly proportional.
"""

import mxnet as mx

class GamblingLoss(mx.gluon.loss.Loss):
    """
    The Gambler's Loss.  See module docstring for details.
    """

    def __init__(self, output: str, threshold=0.001, weight=None, batch_axis=0,
                 include_variance: bool = False, **kwargs):
        """
        Init function.  The +threshold+ parameter indicates the magnitude of
        changes required to create nonzero loss.  The default of 0.001 means
        trades should not be made if changes are under a tenth of a percent.
        The +output+ parameter dictates the format of the prediction tensor.
        If +include_variance+, includes the variance of the loss as well.
        """
        super().__init__(weight, batch_axis, **kwargs)
        self.output = output
        self.threshold = threshold
        self.include_variance = include_variance

    def hybrid_forward(self, F, prediction, target):
        """
        Standard hybrid_forward function.  +prediction+ is the prediction
        tensor, +target+ is the target movement, and +F+ is an mxnet.ndarray
        for use if the network is hybridized.
        """
        assert self.output in ["prediction", "sentiment"]
        if self.output == "prediction":
            zeros = F.zeros(prediction.shape, ctx=prediction.context)

            # Calculate loss for upward motions
            up_loss = F.where(prediction > self.threshold, -target, zeros)
            total_up = F.sum(up_loss, 1, keepdims=True)

            # Calculate loss for downward motions
            down_loss = F.where(prediction < -self.threshold, target, zeros)
            total_down = F.sum(down_loss, 1, keepdims=True)

            # Calculate variance
            if self.include_variance:
                variance = (total_down * total_down) + (total_up * total_up)
                variance = F.sum(variance) / prediction.shape[0]
            else:
                variance = .0

            # Return the total loss and variance
            return (F.sum(total_up) + F.sum(total_down)), variance
        elif self.output == "sentiment":

            # Make prediction based on sentiment and confidence
            up_loss = -prediction[:, 0] * target[:, 0]
            down_loss = prediction[:, 1] * target[:, 0]

            # Calculate variance
            if self.include_variance:
                variance = (up_loss * up_loss) + (down_loss * down_loss)
                variance = F.sum(variance) / prediction.shape[0]
            else:
                variance = .0

            return F.sum(up_loss + down_loss), variance


class GFGLoss(GamblingLoss):
    """
    This is a Gradient Friendly version of the Gambling Loss.  While not 100%
    accurate, it facilitates gradient learning by including the prediction
    value in the final calculation.  Note that this version is designed
    to be lightweight for use with training, and therefore does not
    return the variance.
    """

    def __init__(self, output: str, corrective_l2=0.6, weight=None,
                 batch_axis=0, **kwargs):
        """
        Init function.  The +threshold+ parameter indicates the magnitude of
        changes required to create nonzero loss.  The default of 0.001 means
        trades should not be made if changes are under a tenth of a percent.
        The +output+ parameter dictates the format of the prediction tensor.
        """
        super().__init__(output, weight=weight, batch_axis=batch_axis,
                         include_variance=False, **kwargs)
        self.corrective_l2 = corrective_l2

    def hybrid_forward(self, F, prediction, target):
        """
        Standard hybrid_forward function.  +prediction+ is the predicted
        movement, +target+ is the target movement, and +F+ is an mxnet.ndarray
        for use if the network is hybridized.

        Note that here we use a sigmoid of the prediction value to approximate
        the threshold cutoff.
        """

        if self.output == "prediction":
            # Approximate Gambling Loss with gradient-friendly sigmoid function
            pred_sig = 2 * F.sigmoid(prediction / self.threshold) - 1
            gloss = pred_sig * -target

            # Add corrective L2 loss both to bound correct prediction growth and
            # to provide a gradient for large incorrect predictions
            closs = self.corrective_l2 * F.square(prediction - target)

            return F.sum(gloss + closs)
        else:
            # Other implementations of Gambling Loss are already gradient-
            # friendly
            gfgloss, _ = super().hybrid_forward(F, prediction, target)
            return gfgloss


def find_loss(loss: str, output: str) -> mx.gluon.loss.Loss:
    """
    Returns a loss block associated with the requested +loss+ type, which is
    capable of accepting prediction tensors in the requested +output+ format.
    """

    if loss == "l1":
        if output != "prediction":
            raise RuntimeError(f"L1 Loss not compatible with {output} type")
        return mx.gluon.loss.L1Loss()

    elif loss == "l2":
        if output != "prediction":
            raise RuntimeError(f"L2 Loss not compatible with {output} type")
        return mx.gluon.loss.L2Loss()

    elif loss == "gfg":
        if not output in ["prediction", "sentiment"]:
            raise RuntimeError(f"GFG Loss not compatible with {output} type")
        return GFGLoss(output)

    elif loss == "gambling":
        if not output in ["prediction", "sentiment"]:
            raise RuntimeError(f"Gambling Loss not compatible with {output} type")
        return GamblingLoss(output, include_variance=False)

    elif loss == "gambling-variance":
        if not output in ["prediction", "sentiment"]:
            raise RuntimeError(f"Gambling Loss not compatible with {output} type")
        return GamblingLoss(output, include_variance=True)

    raise RuntimeError(f"{loss} is not a type of Loss block")


