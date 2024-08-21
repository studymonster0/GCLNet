from __future__ import print_function, division
import torch.utils.data
from .modules import *

class GCLNet(nn.Module):

    def __init__(self, in_ch=1, out_ch=1):
        super(GCLNet, self).__init__()

        n1 = 32
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.Maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.Conv1 = conv_block(in_ch, filters[0])
        self.Conv2 = conv_block(filters[0], filters[1])
        self.Conv3 = conv_block(filters[1], filters[2])
        self.Conv4 = conv_block(filters[2], filters[3])
        self.Conv5 = conv_block(filters[3], filters[4])

        self.lconv2 = LFL(filters[1], filters[1])
        self.lconv3 = LFL(filters[2], filters[2])
        self.lconv4 = LFL(filters[3], filters[3])

        self.Up5 = up_conv(filters[4], filters[3])
        self.Up_conv5 = conv_block(filters[4], filters[3])

        self.Up4 = up_conv(filters[3], filters[2])
        self.Up_conv4 = conv_block(filters[3], filters[2])

        self.Up3 = up_conv(filters[2], filters[1])
        self.Up_conv3 = conv_block(filters[2], filters[1])

        self.Up2 = up_conv(filters[1], filters[0])
        self.Up_conv2 = conv_block(filters[1], filters[0])

        self.Conv = nn.Conv2d(filters[0], out_ch, kernel_size=1, stride=1, padding=0)

        self.slice = PGR(8, filters[0])

        self.active = torch.nn.Sigmoid()

        self.glob_grapy = MGR(filters[4], 16)

    def forward(self, x):
        e1 = self.Conv1(x)

        s1 = self.slice(e1)

        e2 = self.Maxpool1(e1)
        e2 = self.Conv2(e2)
        c2 = self.lconv2(e2)

        e3 = self.Maxpool2(c2)
        e3 = self.Conv3(e3)
        c3 = self.lconv3(e3)

        e4 = self.Maxpool3(c3)
        e4 = self.Conv4(e4)
        c4 = self.lconv4(e4)

        e5 = self.Maxpool4(c4)
        e5 = self.Conv5(e5)

        e5 = self.glob_grapy(e5)

        d5 = self.Up5(e5)
        d5 = torch.cat((e4, d5), dim=1)

        d5 = self.Up_conv5(d5)

        d4 = self.Up4(d5)
        d4 = torch.cat((e3, d4), dim=1)
        d4 = self.Up_conv4(d4)

        d3 = self.Up3(d4)
        d3 = torch.cat((e2, d3), dim=1)
        d3 = self.Up_conv3(d3)

        d2 = self.Up2(d3)
        d2 = torch.cat((e1, d2), dim=1)
        d2 = self.Up_conv2(d2)

        d1 = s1 + d2

        out = self.Conv(d1)

        out = self.active(out)

        return out





