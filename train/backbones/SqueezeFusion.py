#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.
# TODO: REFACTOR SQUEEZEFUSION
from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
from backbones import squeezesegV2_quantized
from backbones.squeezesegV2_quantized import BaseNNModule
from torch.quantization import QuantStub, DeQuantStub



class FireImage(BaseNNModule):
    """
    copied from: https://github.com/PRBonn/lidar-bonnetal

    """

    def __init__(self, inplanes, squeeze_planes,
                 expand1x1_planes, expand3x3_planes, bn_d=0.1):
        super(FireImage, self).__init__()
        self.inplanes = inplanes
        self.bn_d = bn_d
        self.activation = nn.ReLU(inplace=False)
        self.squeeze = nn.Conv2d(inplanes, squeeze_planes, kernel_size=1)
        self.squeeze_bn = nn.BatchNorm2d(squeeze_planes, momentum=self.bn_d)
        self.expand1x1 = nn.Conv2d(squeeze_planes, expand1x1_planes,
                                   kernel_size=1)
        self.expand1x1_bn = nn.BatchNorm2d(expand1x1_planes, momentum=self.bn_d)
        self.expand3x3 = nn.Conv2d(squeeze_planes, expand3x3_planes,
                                   kernel_size=3, padding=1)
        self.expand3x3_bn = nn.BatchNorm2d(expand3x3_planes, momentum=self.bn_d)
        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x):
        x = self.activation(self.squeeze_bn(self.squeeze(x)))
        return self.skip_add.cat([
            self.activation(self.expand1x1_bn(self.expand1x1(x))),
            self.activation(self.expand3x3_bn(self.expand3x3(x)))
        ], 1)
    
    def fuse_model(self):
        super().fuse_model()
        torch.quantization.fuse_modules(self,[['squeeze','squeeze_bn'],['expand1x1','expand1x1_bn'],['expand3x3','expand3x3_bn']],inplace=True)


class FireRes(BaseNNModule):
    def __init__(self, inplanes, squeeze_planes, expand1x1_planes, expand3x3_planes, bn_d=0.1):
        super(FireRes, self).__init__()

        self.Fire = FireImage(inplanes, squeeze_planes,
                              expand1x1_planes, expand3x3_planes, bn_d=0.1)
        self.skip = nn.Conv2d(in_channels=inplanes, out_channels=expand3x3_planes + expand1x1_planes, kernel_size=1)
        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, inp):
        x = self.Fire(inp)
        skip = self.skip(inp)
        return self.skip_add.add(x, skip)


class Encoder(BaseNNModule):
    """
    copied from: https://github.com/PRBonn/lidar-bonnetal
    """

    def __init__(self, bn_d, drop_prob):
        # Call the super constructor
        super(Encoder, self).__init__()
        self.bn_d = bn_d
        self.drop_prob = drop_prob

        # last channels
        self.last_channels = 512

        # encoder
        self.fire1 = nn.Sequential(nn.Conv2d(in_channels=64, out_channels=64, kernel_size=2, stride=2, padding=0),
                                   FireRes(64, 16, 64, 64, bn_d=self.bn_d),
                                   FireRes(128, 16, 64, 64, bn_d=self.bn_d))

        self.fire2 = nn.Sequential(nn.Conv2d(in_channels=128, out_channels=128, kernel_size=2, stride=2, padding=0),
                                   FireRes(128, 16, 64, 64, bn_d=self.bn_d),
                                   FireRes(128, 16, 64, 64, bn_d=self.bn_d))
        self.fire3 = nn.Sequential(nn.Conv2d(in_channels=128, out_channels=128, kernel_size=2, stride=2, padding=0),
                                   FireRes(128, 16, 64, 64, bn_d=self.bn_d),
                                   FireRes(128, 16, 64, 64, bn_d=self.bn_d))
        self.fire4 = nn.Sequential(nn.Conv2d(in_channels=128, out_channels=128, kernel_size=2, stride=2, padding=0),
                                   FireRes(128, 32, 128, 128, bn_d=self.bn_d),
                                   FireRes(256, 32, 128, 128, bn_d=self.bn_d))
        self.fire5 = nn.Sequential(nn.Conv2d(in_channels=256, out_channels=256, kernel_size=2, stride=2, padding=0),
                                   FireRes(256, 48, 192, 192, bn_d=self.bn_d),
                                   FireRes(384, 48, 192, 192, bn_d=self.bn_d),
                                   FireRes(384, 64, 256, 256, bn_d=self.bn_d),
                                   FireRes(512, 64, 256, 256, bn_d=self.bn_d))

        # output
        self.dropout = nn.Dropout2d(self.drop_prob)

    def run_layer(self, x, layer, skips, os):
        y = layer(x)
        if y.shape[2] < x.shape[2] or y.shape[3] < x.shape[3]:
            skips[os] = x.detach()
            os *= 2
        x = y
        return x, skips, os

    def forward(self, x):
        # store for skip connections
        skips = {}
        os = 1

        x, skips, os = self.run_layer(x, self.fire1, skips, os)
        x = self.dropout(x)
        # x, skips, os = self.run_layer(x, self.dropout, skips, os)
        x, skips, os = self.run_layer(x, self.fire2, skips, os)
        x = self.dropout(x)

        x, skips, os = self.run_layer(x, self.fire3, skips, os)
        x = self.dropout(x)
        # x, skips, os = self.run_layer(x, self.dropout, skips, os)
        x, skips, os = self.run_layer(x, self.fire4, skips, os)
        x = self.dropout(x)
        # x, skips, os = self.run_layer(x, self.dropout, skips, os)
        x, skips, os = self.run_layer(x, self.fire5, skips, os)
        x = self.dropout(x)
        return x, skips


class Backbone(squeezesegV2_quantized.Backbone):
    """
       Class for Squeezeseg. Subclasses PyTorch's own "nn" module
    """

    def __init__(self, params):
        # Call the super constructor
        super(Backbone, self).__init__(params)
        self.image_encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=False),
            Encoder(bn_d=0.1, drop_prob=0.3),
        )
        self.image_upconv = nn.Sequential(nn.ConvTranspose2d(512, 512,
                                                             kernel_size=[4, 1], stride=[2, 1],
                                                             padding=[1, 0]),
                                          nn.ConvTranspose2d(512, 512,
                                                             kernel_size=[4, 1], stride=[2, 1],
                                                             padding=[1, 0]),
                                          nn.ConvTranspose2d(512, 512,
                                                             kernel_size=[4, 1], stride=[2, 1],
                                                             padding=[1, 0]), )

        self.fusion = nn.Sequential(
            FireRes(1024, 64, 1024, 1024, bn_d=self.bn_d),
            FireRes(2048, 128, 1024, 1024, bn_d=self.bn_d),
            FireRes(2048, 128, 1024, 1024, bn_d=self.bn_d),
            FireRes(2048, 128, 512, 512, bn_d=self.bn_d),
            FireRes(1024, 64, 256, 256, bn_d=self.bn_d),
            FireRes(512, 64, 256, 256, bn_d=self.bn_d)

        )
        self.quant = QuantStub()
        self.dequant = DeQuantStub()
        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x):
        if len(x) == 1:
            return super().forward(x)

        lidar, image = x
        image = self.quant(image)
        image_feat, image_skips = self.image_encoder(image)
        x, skips = super().forward(lidar)
        x = self.quant(x)
        image_feat = self.dequant(image_feat)
        image_feat = self.image_upconv(image_feat)
        image_feat = self.quant(image_feat)
        x = self.skip_add.cat([x, image_feat], dim=1)
        x = self.fusion(x)
        x = self.dequant(x)
        return x, skips

    def fuse_model(self):
        super().fuse_model()
        torch.quantization.fuse_modules(self.image_encoder,['0','1','2'],inplace=True)