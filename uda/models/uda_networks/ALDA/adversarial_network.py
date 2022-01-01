from mmcv.cnn.bricks.wrappers import Linear
import torch.nn as nn
from torch.nn.init import kaiming_uniform_, zeros_, normal_, xavier_normal_
from torch.nn.modules.batchnorm import _BatchNorm
from mmcv.runner import BaseModule

from .utils import calc_coeff, grl_hook


def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find(
            'ConvTranspose2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)


class AdversarialNetwork(BaseModule):

    def __init__(self,
                 in_channels,
                 inner_channels,
                 dropout_rate=0.5,
                 low=0.0,
                 high=1.0,
                 alpha=10,
                 max_iter=10000,
                 init_cfg=None):
        super(AdversarialNetwork, self).__init__(init_cfg=init_cfg)
        self.fc1 = Linear(in_channels, inner_channels)
        self.fc2 = Linear(inner_channels, inner_channels)
        self.fc3 = Linear(inner_channels, 1)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.sigmoid = nn.Sigmoid()
        self.iter_num = 0
        self.low = low
        self.high = high
        self.alpha = alpha
        self.max_iter = max_iter

    def forward(self, x):
        if self.training:
            self.iter_num += 1
        coeff = calc_coeff(self.iter_num, self.high, self.low, self.alpha,
                           self.max_iter)
        x *= 1.0
        x.register_hook(grl_hook(coeff))

        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout2(x)

        x = self.fc3(x)
        out = self.sigmoid(x)

        return out

    def init_weights(self):
        for m in self.modules:
            if isinstance(m, nn.Conv2d):
                kaiming_uniform_(m.weight)
                zeros_(m.bias)
            elif isinstance(m, _BatchNorm):
                normal_(m.weight, 1.0, 0.01)
                zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                xavier_normal_(m.weight)
                zeros_(m.bias)


class MultiAdversarialNetwork(BaseModule):

    def __init__(self,
                 in_channels,
                 inner_channels,
                 num_classes,
                 dropout_rate=0.5,
                 low=0.0,
                 high=1.0,
                 alpha=10,
                 max_iter=10000,
                 init_cfg=None):
        super(AdversarialNetwork, self).__init__(init_cfg=init_cfg)
        self.fc1 = Linear(in_channels, inner_channels)
        self.fc2 = Linear(inner_channels, inner_channels)
        self.fc3 = Linear(inner_channels, num_classes)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.iter_num = 0
        self.low = low
        self.high = high
        self.alpha = alpha
        self.max_iter = max_iter

    def forward(self, x, grl=True):
        if self.training:
            self.iter_num += 1
        if grl and self.training:
            coeff = calc_coeff(self.iter_num, self.high, self.low, self.alpha,
                               self.max_iter)
            x *= 1.0
            x.register_hook(grl_hook(coeff))
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.dropout1(x)

        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout2(x)

        out = self.fc3(x)
        return out
