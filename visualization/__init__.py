import torch

from torchvision.utils import save_image

from model import VDSR
from visualization.cnn_layer_visualization import CNNLayerVisualization


def _filter(net, save_dir, target_conv_index, filter_index):
    """
    summary:
        visualize filter by get activation of random input

    parameters:
        checkpoint_path: path to checkpoint
        save_dir: path to save visualization
        target_conv_index: conv index, start from 1
        filter_index: filter index, start from 1
    """
    if target_conv_index != 'all':
        raise NotImplementedError
    conv_index = 1
    for i, m in enumerate(net.modules()):
        classname = m.__class__.__name__
        if classname.lower().find('conv') != -1:
            layer_vis = CNNLayerVisualization(net, conv_index, i, filter_index - 1, save_dir)
            layer_vis.visualise_layer_with_hooks()
            conv_index += 1