from model.vdsr import VDSR
from model.MSRResNet import MSRResNet, PrunedMSRResNet
from model.CARN import CARN, PrunedCARN
from model.layer import DownSample2DMatlab, UpSample2DMatlab

network_map = {
    'VDSR': VDSR(),
    'PrunedMSRResNet': PrunedMSRResNet(),
    'CARN': CARN(scale=4, group=1),
    'PrunedCARN': PrunedCARN(multi_scale=4, group=1)
}

def get_network(name):
    if name in network_map.keys():
        return network_map[name]
    else:
        raise ValueError(f"{name} is not in the network_map")