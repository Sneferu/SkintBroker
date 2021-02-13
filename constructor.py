"""
Constructor

This module is responsible for building components based on passed parameters.
It consists of a series of functions for building each type of SkintBroker
structure, each of which accepts a dictionary of parameters that dovetail
with the info file structure.
"""

from typing import Any, Dict, List

from . import models, presenters, providers
from .parser import safe_get


# Dictionary for assigning nets to type specifiers
net_types = {
        "textcnn": models.TextCNNNet,
        "conv": models.DailyConvolutionalNet,
        "rnn": models.DailyRecurrentNet,
        "dense": models.DenseNet,
        "macd": models.technical.MACDBlock,
        "volume": models.technical.VolumeBlock,
        "momentum": models.technical.MomentumBlock,
        "mass_index": models.technical.MassIndexBlock,
        "trix": models.technical.TRIXBlock,
        "vortex": models.technical.VortexBlock,
        "rsi": models.technical.RSIBlock,
        "stochastic": models.technical.StochasticOscillatorBlock,
        "williams": models.technical.WilliamsBlock,
        "accdist": models.technical.AccDistBlock,
        "mfi": models.technical.MoneyFlowIndexBlock,
        "vpt": models.technical.VolumePriceTrendBlock,
        "obv": models.technical.OnBalanceVolumeBlock,
        "dysart": models.technical.DysartBlock,
        "donchian": models.technical.DonchianChannelBlock,
        "bollinger_breakout": models.technical.BollingerBreakoutBlock,
        "bollinger_bounce": models.technical.BollingerBounceBlock,
        "ultimate": models.technical.UltimateOscillatorBlock,
        "target": models.technical.TargetBlock,
        "concat": models.structural.ConcatenateNet
        }


# Dictionary for assigning models to type specifiers
model_types = {
        "sequential": models.SequentialModel,
        "recurrent": models.RecurrentModel
        }


class ContainerNet(models.Net):
    """
    Net for containing other nets
    """
    # Only one public method is needed
    # pylint: disable=too-few-public-methods

    def __init__(self, nets: Dict[str, Any], data_features: List[str],
                 **kwargs) -> None:
        """
        Init function.  Takes a list of net info structures +nets+ and a
        list of input features +data_features+.
        """
        super().__init__(data_features, **kwargs)

        # Set up maintenance dictionaries
        self.inputs_by_block = {}
        self.hiddens_by_block = {}
        with self.name_scope():

            # First, generate and store blocks.
            nets_by_name = {}
            for net_name, net in nets.items():

                # Get name and type
                net_type = safe_get(net, "type")

                # Get all input nets and determine their features
                input_features = []
                input_blocks = []
                for input_name in safe_get(net, "inputs"):

                    # First check against the presenter
                    if input_name == "<presenter>":
                        input_features.append(data_features)
                        input_blocks.append(None)
                        continue

                    # Next, get the corresponding net and exract features
                    if not input_name in nets_by_name:
                        raise RuntimeError(f"Net {input_name} not yet declared")
                    input_net = nets_by_name[input_name]
                    input_blocks.append(input_net)
                    input_features.append(input_net.features)

                # Finally, construct the net and add it to dictionaries
                params = net.get("params", {})
                net_class = safe_get(net_types, net_type)
                net_block = net_class(input_features, **params)
                self.register_child(net_block)
                nets_by_name[net_name] = net_block
                self.inputs_by_block[net_block] = input_blocks

    def forward(self, inputs):
        """
        Returns the outputs of the net.
        """
        # For each block in the list
        outputs_by_block = {None: inputs}
        last_output = None
        for block in self.inputs_by_block:

            # First, gather inputs
            ins = [outputs_by_block[iblock] \
                   for iblock in self.inputs_by_block[block]]

            # Next, find any hidden states and evaluate
            state = self.hiddens_by_block.get(block, None)
            if state is not None:
                output_tuple = block(*ins, *state)
                self.hiddens_by_block[block] = [*output_tuple[1:]]
                output = output_tuple[0]
            else:
                output = block(*ins)

            # Finally, add the output to the output list
            outputs_by_block[block] = output
            last_output = output

        # Return the final output
        return last_output

    @property
    def features(self) -> List[str]:
        """
        A list of features output by this net.
        """
        # Return the feature set from the last block of the net
        return list(self.inputs_by_block.keys())[-1].features

    @property
    def trainable(self) -> bool:
        """
        Whether or not this net is trainable
        """
        # If any block is trainable, so is the net
        for block in self.inputs_by_block:
            if block.trainable:
                return True
        return False

    def begin_state(self, **kwargs: Dict[str, Any]) -> List[Any]:
        """
        Performs state initialization.
        """
        states = []
        for block in self.inputs_by_block:
            state = block.begin_state(**kwargs)
            if state:
                self.hiddens_by_block[block] = state
                states.extend(state)

        return states


def build_model(ticker: str, name: str, params: Dict[str, Any]) \
        -> models.Model:
    """
    Builds a model called +name+ for a +ticker+ from a dictionary of +params+.
    """

    # First get the requested data provider and presenter
    prov_info = safe_get(params, "provider")
    provider = build_provider(ticker, prov_info)
    pres_info = safe_get(params, "presenter")
    presenter = build_presenter(provider, pres_info)

    # Next, build a net which contains all requested nets
    nets_info = safe_get(params, "nets")
    net = ContainerNet(nets_info, presenter.data_features())

    # Finally, construct a model for training said net
    model_type = safe_get(model_types, safe_get(params, "type"))

    # Don't expose a net directly to a raw gambling loss.  Give it the
    # gradient-friendly approximation
    loss_type = safe_get(params, "loss")
    loss_type = "gfg" if loss_type == "gambling" else loss_tyle
    loss = models.find_loss(loss_type, safe_get(params, "output"))

    model = model_type(net, [presenter], name, loss=loss,
                       **safe_get(params, "params"))

    return model


def build_presenter(provider: providers.DataProvider,
                    params: Dict[str, Any]) -> presenters.DataPresenter:
    """
    Builds a data presenter from a dictionary of +params+ and a +ticker+,
    pulling data from a DataProvider.
    """
    types = {"intraday": presenters.IntradayPresenter}

    # First, gather presenter type and parameters
    pres = safe_get(params, "type")
    pres_type = safe_get(types, pres)
    pres_params = safe_get(params, "params")

    # Finally, construct and return the presenter
    return pres_type(provider, **pres_params)


def build_provider(ticker: str, params: Dict[str, Any]) \
        -> providers.DataProvider:
    """
    Builds a data provider from a dictionary of +params+ and a +ticker+.
    """
    types = {"alphavantage": providers.AVDataProvider}

    # First, gather provider type and parameters
    prov = params.get("type")
    prov_type = safe_get(types, prov)
    prov_params = safe_get(params, "params")

    # Then construct and return the provider
    return prov_type(ticker, **prov_params)
