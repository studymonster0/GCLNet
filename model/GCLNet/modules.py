from __future__ import print_function, division
import torch.utils.data
from .utils import *

class MGR(nn.Module):

    def __init__(self, inch, N=16):
        super(MGR, self).__init__()

        self.N = N

        self.conv11 = single_conv(inch, inch, 3, 1)
        self.conv12 = nn.Sequential(
            nn.Conv2d(inch, inch, 3, 1, dilation=3, padding=3),
            nn.BatchNorm2d(inch),
            nn.ReLU(inplace=True)
        )
        self.conv13 = nn.Sequential(
            nn.Conv2d(inch, inch, 3, 1, dilation=5, padding=5),
            nn.BatchNorm2d(inch),
            nn.ReLU(inplace=True)
        )

        self.convN = nn.Conv2d(inch, N, 1, 1, padding=0, bias=False)

        self.conv2 = nn.Sequential(
            nn.Conv1d(3 * N, N, 1, 1, bias=False),
            nn.Conv1d(N, 3 * N, 1, 1, bias=False),
            nn.ReLU(inplace=True)
        )

        self.conv21 = nn.Sequential(
            nn.Conv1d(N, N, 1, 1, bias=False),
            nn.Linear(inch, inch, bias=False),
            nn.ReLU(inplace=True)
        )
        self.conv22 = nn.Sequential(
            nn.Conv1d(N, N, 1, 1, bias=False),
            nn.Linear(inch, inch, bias=False),
            nn.ReLU(inplace=True)
        )
        self.conv23 = nn.Sequential(
            nn.Conv1d(N, N, 1, 1, bias=False),
            nn.Linear(inch, inch, bias=False),
            nn.ReLU(inplace=True)
        )

        self.conv31 = nn.Sequential(
            nn.Conv1d(N, N, 1, 1, bias=False),
            nn.Linear(inch, inch, bias=False),
            nn.ReLU(inplace=True)
        )
        self.conv32 = nn.Sequential(
            nn.Conv1d(N, N, 1, 1, bias=False),
            nn.Linear(inch, inch, bias=False),
            nn.ReLU(inplace=True)
        )
        self.conv33 = nn.Sequential(
            nn.Conv1d(N, N, 1, 1, bias=False),
            nn.Linear(inch, inch, bias=False),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        B, C, H, W = x.shape

        x10 = self.conv11(x)
        x20 = self.conv12(x)
        x30 = self.conv13(x)

        x1 = x10.view(B, C, H * W)
        x2 = x20.view(B, C, H * W)
        x3 = x30.view(B, C, H * W)

        w = self.convN(x).view(B, self.N, H * W)  # B N H*W

        xN1 = self.conv21(torch.bmm(w, torch.transpose(x1, 1, 2)))  # B N C
        xN2 = self.conv22(torch.bmm(w, torch.transpose(x2, 1, 2)))  # B N C
        xN3 = self.conv23(torch.bmm(w, torch.transpose(x3, 1, 2)))  # B N C

        xN = torch.cat([xN1, xN2, xN3], dim=1)  # B 3N C
        xNout = self.conv2(xN)  # B N C

        y1 = self.conv31(xNout[:, :self.N, :])  # B N C
        y2 = self.conv32(xNout[:, self.N: (self.N * 2), :])  # B N C
        y3 = self.conv33(xNout[:, (self.N * 2): (self.N * 3), :])  # B N C

        out1 = torch.bmm(torch.transpose(y1, 1, 2), w).view(B, C, H, W)
        out2 = torch.bmm(torch.transpose(y2, 1, 2), w).view(B, C, H, W)
        out3 = torch.bmm(torch.transpose(y3, 1, 2), w).view(B, C, H, W)

        out = out1 + out2 + out3
        return out

class LFL(nn.Module):
    def __init__(self, in_ch=1, out_ch=1):
        super(LFL, self).__init__()

        self.conv31 = nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=1, padding=1)
        self.conv32 = nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=1, padding=3, dilation=3)
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

        self.conv11 = nn.Sequential(
            nn.Conv1d(in_ch, in_ch // 16, kernel_size=1, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_ch // 16, in_ch, kernel_size=1, stride=1),
            nn.Softmax(dim=1)
        )

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

        self.globalavg = nn.AdaptiveAvgPool2d((1, 1))

    def selfattention(self, x3):
        B, C, H, W = x3.shape
        L = H * W

        x3 = self.globalavg(x3).view(B, -1, 1)

        weight = self.conv11(x3).unsqueeze(-1)

        return weight

    def forward(self, x):
        B, C, H, W = x.shape
        L = H * W

        x1 = self.conv31(x)
        x2 = self.conv32(x)

        x3 = self.maxpool3(x)
        weight = self.selfattention(x3)

        y = x1 + x2
        y = self.conv1(y)

        out = weight * y
        out = self.conv2(out)
        out = out + x

        return out

class PGR(nn.Module):

    def __init__(self, p2=4, nIn=32, nOut=32, add=True):
        super(PGR, self).__init__()
        self.p2 = p2
        self.N = nIn // 4

        self.add = add

        self.graph0 = graph(p2, nIn, self.N)

        self.conv31 = nn.Sequential(
            nn.Conv2d(nOut, nOut, kernel_size=1, stride=1),
            nn.BatchNorm2d(nOut),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        b, C, H, W = x.shape

        h, w = H // self.p2, W // self.p2
        L = h * w

        # 分块
        fs = torch.zeros((b, self.p2 ** 2, C, h, w)).cuda()  # 存储小块 B×16×C×h×w
        for i in range(1, self.p2 + 1):
            for j in range(1, self.p2 + 1):
                fs[:, i * j - 1, :, :, :] = x[:, :, (i - 1) * h: i * h, (j - 1) * w: j * w]
        fs = fs.view(b * self.p2 ** 2, C, h, w)  # (B×16)×C×h×w

        fs6 = self.graph0(fs, x)

        # 拼接
        out = torch.zeros_like(x)
        for i in range(1, self.p2 + 1):
            for j in range(1, self.p2 + 1):
                out[:, :, (i - 1) * h: i * h, (j - 1) * w: j * w] = fs6[:, i * j - 1, :, :, :]

        out = self.conv31(out)

        if self.add:
            out = out + x

        return out


class graph(nn.Module):

    def __init__(self, p2=4, nIn=64, N=16):
        super(graph, self).__init__()
        self.p2 = p2
        self.N = N

        self.conv30 = nn.Sequential(
            nn.Conv2d(nIn, self.N, kernel_size=3, stride=1, padding=1, groups=1),
            nn.ReLU(inplace=True)
        )

        self.conv10 = nn.Sequential(
            nn.Conv1d(nIn, nIn, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True)
        )

        self.conv11 = nn.Sequential(
            nn.Conv1d(self.N, self.N, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True)
        )

        self.adaptivemax = nn.AdaptiveAvgPool2d((8, 8))

        self.conv12 = nn.Sequential(
            nn.Conv1d(p2 ** 2, p2, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv1d(p2, p2, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv1d(p2, p2 ** 2, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )

    def ADP_weight(self, x):
        b, C, H, W = x.shape

        fg = self.adaptivemax(x)  # B×C×4×4
        fg1 = fg.view(b, C, self.p2 ** 2)  # B×C×16
        fg1 = torch.transpose(fg1, 1, 2)  # B×16×C
        fg2 = self.conv12(fg1)  # B×16×C
        fg3 = fg2.unsqueeze(-1).unsqueeze(-1)

        return fg3

    def graph_convolution(self, fs, x):
        b, C, H, W = x.shape
        h, w = H // self.p2, W // self.p2
        L = h * w

        B = self.conv30(fs)  # (B×16)×N×h×w
        B1 = B.view(-1, self.N, L)  # (B×16)×N×L

        fs1 = fs.view(-1, C, L)  # (B×16)×C×L
        fs1 = torch.transpose(fs1, 1, 2)  # (B×16)×L×C

        fs2 = torch.bmm(B1, fs1)  # (B×16)×N×C

        fs3 = self.conv11(fs2)  # (B×16)×N×C

        # fs4 = fs2 - fs3  # (B×16)×N×C

        fs5 = self.conv10(torch.transpose(fs3, 1, 2))  # (B×16)×C×N

        fs6 = torch.bmm(torch.transpose(B1, 1, 2), torch.transpose(fs5, 1, 2))  # (B×16)×L×C
        fs6 = torch.transpose(fs6, 1, 2)  # (B×16)×C×L
        fs6 = fs6.view(b, self.p2 ** 2, C, h, w)  # B×16×C×h×w

        return fs6

    def forward(self, fs, x):
        fs6 = self.graph_convolution(fs, x)

        weight = self.ADP_weight(x)

        out = weight * fs6

        return out