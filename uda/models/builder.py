from mmcv.runner import BaseModule
from mmcv.cnn import MODELS as MMCV_MODELS
from mmcv.utils import Registry

MODELS = Registry('models', parent=MMCV_MODELS)
UDA_NETWORK = MODELS
LOSSES = MODELS


def build_uda_network(cfg: dict) -> BaseModule:
    return UDA_NETWORK.build(cfg)


def build_uda_loss(cfg: dict) -> BaseModule:
    return LOSSES.build(cfg)
