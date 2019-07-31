# This file was modified from https://github.com/BobLiu20/YOLOv3_PyTorch
# It needed to be modified in order to accomodate for different strides in the

import torch.nn as nn
from collections import OrderedDict
import torch.nn.functional as F


class BasicBlock(nn.Module):
  def __init__(self, inplanes, planes, bn_d=0.1):
    super(BasicBlock, self).__init__()
    self.conv1 = nn.Conv2d(inplanes, planes[0], kernel_size=1,
                           stride=1, padding=0, bias=False)
    self.bn1 = nn.BatchNorm2d(planes[0], momentum=bn_d)
    self.relu1 = nn.LeakyReLU(0.1)
    self.conv2 = nn.Conv2d(planes[0], planes[1], kernel_size=3,
                           stride=1, padding=1, bias=False)
    self.bn2 = nn.BatchNorm2d(planes[1], momentum=bn_d)
    self.relu2 = nn.LeakyReLU(0.1)

  def forward(self, x):
    residual = x

    out = self.conv1(x)
    out = self.bn1(out)
    out = self.relu1(out)

    out = self.conv2(out)
    out = self.bn2(out)
    out = self.relu2(out)

    out += residual
    return out


# ******************************************************************************

class Decoder(nn.Module):
  """
     Class for DarknetSeg. Subclasses PyTorch's own "nn" module
  """

  def __init__(self, params, stub_skips, OS=32, feature_depth=1024):
    super(Decoder, self).__init__()
    self.backbone_OS = OS
    self.backbone_feature_depth = feature_depth
    self.drop_prob = params["dropout"]
    self.bn_d = params["bn_d"]

    # stride play
    self.strides = [2, 2, 2, 2, 2]
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
    self.dec5 = self._make_dec_layer(BasicBlock,
                                     [self.backbone_feature_depth, 512],
                                     bn_d=self.bn_d,
                                     stride=self.strides[0])
    self.dec4 = self._make_dec_layer(BasicBlock, [512, 256], bn_d=self.bn_d,
                                     stride=self.strides[1])
    self.dec3 = self._make_dec_layer(BasicBlock, [256, 128], bn_d=self.bn_d,
                                     stride=self.strides[2])
    self.dec2 = self._make_dec_layer(BasicBlock, [128, 64], bn_d=self.bn_d,
                                     stride=self.strides[3])
    self.dec1 = self._make_dec_layer(BasicBlock, [64, 32], bn_d=self.bn_d,
                                     stride=self.strides[4])

    # layer list to execute with skips
    self.layers = [self.dec5, self.dec4, self.dec3, self.dec2, self.dec1]

    # for a bit of fun
    self.dropout = nn.Dropout2d(self.drop_prob)

    # last channels
    self.last_channels = 32

  def _make_dec_layer(self, block, planes, bn_d=0.1, stride=2):
    layers = []

    #  downsample
    if stride == 2:
      layers.append(("upconv", nn.ConvTranspose2d(planes[0], planes[1],
                                                  kernel_size=[1, 4], stride=[1, 2],
                                                  padding=[0, 1])))
    else:
      layers.append(("conv", nn.Conv2d(planes[0], planes[1],
                                       kernel_size=3, padding=1)))
    layers.append(("bn", nn.BatchNorm2d(planes[1], momentum=bn_d)))
    layers.append(("relu", nn.LeakyReLU(0.1)))

    #  blocks
    layers.append(("residual", block(planes[1], planes, bn_d)))

    return nn.Sequential(OrderedDict(layers))

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
    x, skips, os = self.run_layer(x, self.dec5, skips, os)
    x, skips, os = self.run_layer(x, self.dec4, skips, os)
    x, skips, os = self.run_layer(x, self.dec3, skips, os)
    x, skips, os = self.run_layer(x, self.dec2, skips, os)
    x, skips, os = self.run_layer(x, self.dec1, skips, os)

    x = self.dropout(x)

    return x

  def get_last_depth(self):
    return self.last_channels
