
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import copy
import types

from . import measure
from ..p_utils import get_layer_metric_array

from neutrino.framework.torch_model import TorchModel
from neutrino.core.lgetter import LayersGetter
from deeplite.torch_profiler.torch_data_loader import TorchForwardPass
from neutrino.framework.nn import Linear, BaseConvolution




@measure("min_depth")
def min_depth_measure(net, inputs, targets, *args, **kwargs):
    """Returns the length of the shortest path from input to output"""
    nt_model = TorchModel(net, (inputs.cpu(),))
    graph = nt_model.parse_into_graph()
    node_views = graph.distance_from_outputs_view((Linear, BaseConvolution), grouped=True,
                                                              shortest_path=True, reverse=True,
                                                              include_output_nodes=True)
    depth = node_views[0][0].attributes['dtn']
    return depth
    # breakpoint()
    # g = nt_model.parse_into_graph()
    # fp = TorchForwardPass(model_input_pattern=(0, '_'))
    # lgetter = LayersGetter.factory('g_dfo_not')
    # layers = lgetter.get_from_model(nt_model)
    # breakpoint()
    # return len(layers)


@measure("max_depth")
def max_depth_measure(net, inputs, targets, *args, **kwargs):
    """Returns the length of the shortest path from input to output"""
    nt_model = TorchModel(net, (inputs.cpu(),))
    graph = nt_model.parse_into_graph()
    node_views = graph.distance_from_outputs_view((Linear, BaseConvolution), grouped=True,
                                                              shortest_path=False, reverse=True,
                                                              include_output_nodes=True)
    depth = node_views[0][0].attributes['dtn']
    return depth


# def contains_depthwise_conv(net, inputs, targets, *args, **kwargs):
#     nt_model = TorchModel(net, (inputs.cpu(),))
#     # graph = nt_model.parse_into_graph()

#     layers = LayersGetter.factory('ill_il_ig')
#     has_depthwise = any([l.groups > 1 for l in layers if isinstance(l, BaseConvolution)])
#     return float(has_depthwise)