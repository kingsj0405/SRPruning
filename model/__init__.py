from model.vdsr import VDSR
from model.MSRResNet import MSRResNet, PrunedMSRResNet
from model.CARN import CARN
from model.srdensenet import Net as SRDenseNet
from model.rdn import RDN
from model.RRDBNet_arch import RRDBNet
from model.RCAN import RCAN
from model.layer import DownSample2DMatlab, UpSample2DMatlab

network_map = {
    'VDSR': VDSR(),
    'MSRResNet': MSRResNet(),
    'PrunedMSRResNet': PrunedMSRResNet(),
    'CARN': CARN(scale=4, group=1),
    'PCARN6': CARN(multi_scale=4, group=1, channel_cnt=6),
    'PCARN12': CARN(multi_scale=4, group=1, channel_cnt=12),
    'PCARN18': CARN(multi_scale=4, group=1, channel_cnt=18),
    'PCARN32': CARN(multi_scale=4, group=1, channel_cnt=32),
    'SRDenseNet': SRDenseNet(),
    'RDN': RDN(scale_factor=4, num_channels=3, num_features=64,
               growth_rate=64, num_blocks=16, num_layers=8),
    'RRDB': RRDBNet(3, 3, 64, 23, gc=32),
    'RCAN': RCAN
}

def get_network(name):
    if name in network_map.keys():
        return network_map[name]
    else:
        raise ValueError(f"{name} is not in the network_map")