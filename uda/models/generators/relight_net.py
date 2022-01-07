import torch.utils.checkpoint as cp
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmcv.cnn.bricks.norm import build_norm_layer
from mmcv.runner import BaseModule, Sequential


class ResNetBasicBlock(BaseModule):
    """Basic block for ResNet."""

    expansion = 1

    def __init__(self,
                 in_channels,
                 channels,
                 stride=1,
                 dilation=1,
                 with_cp=False,
                 dropout_rate=0.5,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU', inplace=False),
                 init_cfg=None):
        super(ResNetBasicBlock, self).__init__(init_cfg)

        self.conv1 = ConvModule(in_channels,
                                channels,
                                kernel_size=3,
                                stride=stride,
                                padding=dilation,
                                dilation=dilation,
                                norm_cfg=norm_cfg,
                                act_cfg=act_cfg)
        self.use_dropout = True if dropout_rate > 0 else False
        if self.use_dropout:
            self.dropout = nn.Dropout(dropout_rate)
        self.conv2 = ConvModule(channels,
                                channels,
                                kernel_size=3,
                                padding=1,
                                norm_cfg=norm_cfg,
                                act_cfg=act_cfg)

        self.relu = nn.ReLU()
        self.stride = stride
        self.dilation = dilation
        self.with_cp = with_cp

    def forward(self, x):
        """Forward function."""

        def _inner_forward(x):
            identity = x

            x = self.conv1(x)
            if self.use_dropout:
                out = self.dropout(x)
            out = self.conv2(x)
            out = out + identity

            return out

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)

        out = self.relu(out)

        return out


class LightImgGenerator(BaseModule):

    def __init__(self,
                 in_channels,
                 out_channels,
                 base_channels=64,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 num_downsampling=2,
                 dropout_rate=0.5,
                 num_blocks=6,
                 init_cfg=None):
        assert num_blocks > 0
        super(LightImgGenerator, self).__init__(init_cfg)

        # downsample
        downsample = [
            nn.ReflectionPad2d(3),
            ConvModule(in_channels,
                       base_channels,
                       kernel_size=7,
                       padding=0,
                       norm_cfg=norm_cfg,
                       act_cfg=act_cfg)
        ]
        for i in range(num_downsampling):
            rate = 2**i
            downsample_layer = ConvModule(base_channels * rate,
                                          base_channels * rate * 2,
                                          kernel_size=3,
                                          stride=2,
                                          padding=1,
                                          norm_cfg=norm_cfg,
                                          act_cfg=act_cfg)
            downsample.append(downsample_layer)
        self.downsample_blocks = Sequential(*downsample)

        # feature extraction
        rate = 2**num_downsampling
        blocks = []
        for i in range(num_blocks):
            blocks.append(
                ResNetBasicBlock(base_channels * rate,
                                 base_channels * rate,
                                 dropout_rate=dropout_rate))
        self.res_blocks = Sequential(*blocks)

        # upsample
        upsample = []
        for i in range(num_downsampling):
            rate = 2**(num_downsampling - i)
            deconv = nn.ConvTranspose2d(base_channels * rate,
                                        int(base_channels * rate / 2),
                                        kernel_size=3,
                                        stride=2,
                                        padding=1,
                                        output_padding=1,
                                        bias=False)
            norm = build_norm_layer(norm_cfg, int(base_channels * rate / 2))[1]
            upsample.append(Sequential(deconv, norm, nn.ReLU()))
        self.upsample_blocks = Sequential(*upsample)

        # output
        self.output = Sequential(
            nn.ReflectionPad2d(3),
            ConvModule(base_channels,
                       out_channels,
                       kernel_size=7,
                       padding=0,
                       act_cfg=None), nn.Tanh())

    def forward(self, inputs):
        x = self.downsample_blocks(inputs)
        x = self.res_blocks(x)
        x = self.upsample_blocks(x)
        out = self.output(x)
        return out


class FCDiscriminator(BaseModule):

    def __init__(self,
                 num_classes,
                 base_channels=64,
                 num_convs=4,
                 init_cfg=None):
        super(FCDiscriminator, self).__init__(init_cfg)
        self.conv1 = nn.Conv2d(num_classes,
                               base_channels,
                               kernel_size=4,
                               stride=2,
                               padding=1)
        self.conv2 = nn.Conv2d(base_channels,
                               base_channels * 2,
                               kernel_size=4,
                               stride=2,
                               padding=1)
        self.conv3 = nn.Conv2d(base_channels * 2,
                               base_channels * 4,
                               kernel_size=4,
                               stride=1,
                               padding=1)
        self.conv4 = nn.Conv2d(base_channels * 4,
                               base_channels * 4,
                               kernel_size=4,
                               stride=1,
                               padding=1)
        self.classifier = nn.Conv2d(base_channels * 4,
                                    1,
                                    kernel_size=4,
                                    stride=1,
                                    padding=1)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.leaky_relu(x)
        x = self.conv2(x)
        x = self.leaky_relu(x)
        x = self.conv3(x)
        x = self.leaky_relu(x)
        x = self.conv4(x)
        x = self.leaky_relu(x)
        x = self.classifier(x)
        return x
