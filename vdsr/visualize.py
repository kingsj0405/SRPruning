import fire
import torch

from torchvision.utils import save_image

from model import VDSR
from visualization.cnn_layer_visualization import CNNLayerVisualization


def _filter(net, save_dir, layer_index, filter_index):
    print(f"[INFO] Process Conv-{layer_index}.{filter_index}")
    conv_index = []
    for i, m in enumerate(net.modules()):
        classname = m.__class__.__name__
        if classname.lower().find('conv') != -1:
            conv_index.append(i)
    layer_vis = CNNLayerVisualization(net, conv_index[layer_index], filter_index, save_dir)
    layer_vis.visualise_layer_with_hooks()


def filter(checkpoint_path, save_dir, layer_index, filter_index):
    print(f"[INFO] Load checkpoitn from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path)
    net = VDSR()
    net.load_state_dict(checkpoint['net'])
    _filter(net, save_dir, layer_index, filter_index)


if __name__ == '__main__':
    fire.Fire({
        'filter': filter
    })