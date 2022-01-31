import torch
import torch.nn as nn


class EnhanceBranch(nn.Module):
    """Image Contrast Enhancement Network"""
    def __init__(self, in_channels=3, channels=64):
        super().__init__()
        self.downsample = nn.Conv2d(in_channels=in_channels,
                                    out_channels=channels,
                                    kernel_size=9,
                                    padding=4)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.conv1 = nn.Conv2d(in_channels=channels,
                               out_channels=channels,
                               kernel_size=3,
                               padding=1)
        self.conv2 = nn.Conv2d(in_channels=channels,
                               out_channels=channels * 2,
                               kernel_size=3)
        self.conv3 = nn.Conv2d(in_channels=channels * 2,
                               out_channels=channels * 2,
                               kernel_size=3,
                               padding=1)
        self.conv4 = nn.ConvTranspose2d(in_channels=channels * 2,
                                        out_channels=channels,
                                        kernel_size=3)
        self.conv5 = nn.Conv2d(in_channels=channels * 2,
                               out_channels=channels,
                               kernel_size=3,
                               padding=1)
        self.conv6 = nn.Conv2d(in_channels=(channels + channels // 2),
                               out_channels=channels,
                               kernel_size=3,
                               padding=1)
        self.out_conv = nn.Conv2d(in_channels=channels,
                                  out_channels=4,
                                  kernel_size=3,
                                  padding=1)
        self.sub_conv = nn.Conv2d(in_channels=in_channels,
                                  out_channels=channels // 2,
                                  kernel_size=3,
                                  padding=1)

    def forward(self, inputs):
        x = self.downsample(inputs)
        sub_x = self.sub_conv(inputs)
        x1 = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x1))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = torch.cat([x, x1], dim=1)
        x = self.relu(self.conv5(x))
        x = torch.cat([x, sub_x], dim=1)
        x = self.conv6(x)
        x = self.out_conv(x)
        out = self.sigmoid(x)
        return out


class ImageEnhancementtNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.ice_net = EnhanceBranch()
        self.red_net = EnhanceBranch(in_channels=4)

    def forward(self, inputs):
        ice_out = self.ice_net(inputs)
        max_chanel_map = torch.max(ice_out, dim=1).values.unsqueeze(1)
        x = torch.cat([inputs, max_chanel_map], dim=1)
        red_out = self.red_net(x)
        return tuple([ice_out, red_out, inputs])