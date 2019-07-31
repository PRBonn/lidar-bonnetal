#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.

from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F


class FireUp(nn.Module):

  def __init__(self, inplanes, squeeze_planes,
               expand1x1_planes, expand3x3_planes, bn_d, stride):
    super(FireUp, self).__init__()
    self.inplanes = inplanes
    self.bn_d = bn_d
    self.stride = stride
    self.activation = nn.ReLU(inplace=True)
    self.squeeze = nn.Conv2d(inplanes, squeeze_planes, kernel_size=1)
    self.squeeze_bn = nn.BatchNorm2d(squeeze_planes, momentum=self.bn_d)
    if self.stride == 2:
      self.upconv = nn.ConvTranspose2d(squeeze_planes, squeeze_planes,
                                       kernel_size=[1, 4], stride=[1, 2],
                                       padding=[0, 1])
    self.expand1x1 = nn.Conv2d(squeeze_planes, expand1x1_planes,
                               kernel_size=1)
    self.expand1x1_bn = nn.BatchNorm2d(expand1x1_planes, momentum=self.bn_d)
    self.expand3x3 = nn.Conv2d(squeeze_planes, expand3x3_planes,
                               kernel_size=3, padding=1)
    self.expand3x3_bn = nn.BatchNorm2d(expand3x3_planes, momentum=self.bn_d)

  def forward(self, x):
    x = self.activation(self.squeeze_bn(self.squeeze(x)))
    if self.stride == 2:
      x = self.activation(self.upconv(x))
    return torch.cat([
        self.activation(self.expand1x1_bn(self.expand1x1(x))),
        self.activation(self.expand3x3_bn(self.expand3x3(x)))
    ], 1)


# ******************************************************************************

class Decoder(nn.Module):
  """
     Class for DarknetSeg. Subclasses PyTorch's own "nn" module
  """

  def __init__(self, params, stub_skips, OS=32, feature_depth=512):
    super(Decoder, self).__init__()
    self.backbone_OS = OS
    self.backbone_feature_depth = feature_depth
    self.drop_prob = params["dropout"]
    self.bn_d = params["bn_d"]

    # stride play
    self.strides = [2, 2, 2, 2]
    # check current stride
    current_os = 1
    for s in self.strides:
      current_os *= s
    print("Decoder original OS: ", int(current_os))
    # redo strides according to needed stride
    for i, stride in enumerate(self.strides):
      if int(current_os) != self.backbone_OS:
        if stride == 2:
          current_os /= 2
          self.strides[i] = 1
        if int(current_os) == self.backbone_OS:
          break
    print("Decoder new OS: ", int(current_os))
    print("Decoder strides: ", self.strides)

    # decoder
    # decoder
    self.firedec10 = FireUp(self.backbone_feature_depth,
                            64, 128, 128, bn_d=self.bn_d,
                            stride=self.strides[0])
    self.firedec11 = FireUp(256, 32, 64, 64, bn_d=self.bn_d,
                            stride=self.strides[1])
    self.firedec12 = FireUp(128, 16, 32, 32, bn_d=self.bn_d,
                            stride=self.strides[2])
    self.firedec13 = FireUp(64, 16, 32, 32, bn_d=self.bn_d,
                            stride=self.strides[3])

    # for a bit of fun
    self.dropout = nn.Dropout2d(self.drop_prob)

    # last channels
    self.last_channels = 64

  def run_layer(self, x, layer, skips, os):
    feats = layer(x)  # up
    if feats.shape[-1] > x.shape[-1]:
      os //= 2  # match skip
      feats = feats + skips[os].detach()  # add skip
    x = feats
    return x, skips, os

  def forward(self, x, skips):
    os = self.backbone_OS

    # run layers
    x, skips, os = self.run_layer(x, self.firedec10, skips, os)
    x, skips, os = self.run_layer(x, self.firedec11, skips, os)
    x, skips, os = self.run_layer(x, self.firedec12, skips, os)
    x, skips, os = self.run_layer(x, self.firedec13, skips, os)

    x = self.dropout(x)

    return x

  def get_last_depth(self):
    return self.last_channels
