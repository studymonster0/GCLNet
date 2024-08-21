from math import sqrt
import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.nn.functional as F
from utils import *
import os
from loss import *
from model import *
from skimage.feature.tests.test_orb import img

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

class Net(nn.Module):
    def __init__(self, model_name, mode):
        super(Net, self).__init__()
        self.model_name = model_name
        self.dice_loss = dice_coeff()
        self.focal_loss = Focal_Loss()
        self.turskey_loss = TverskyLoss()
        # self.gamma = nn.Parameter(torch.ones(1, dtype=torch.float32) * 0.5, requires_grad=True)
        self.cal_loss = SoftIoULoss()
        if model_name == 'DNANet':
            if mode == 'train':
                self.model = DNANet(mode='train')
            else:
                self.model = DNANet(mode='test')  
        elif model_name == 'DNANet_BY':
            if mode == 'train':
                self.model = DNAnet_BY(mode='train')
            else:
                self.model = DNAnet_BY(mode='test')  
        elif model_name == 'ACM':
            self.model = ACM()
        elif model_name == 'ALCNet':
            self.model = ALCNet()
        elif model_name == 'ISNet':
            if mode == 'train':
                self.model = ISNet(mode='train')
            else:
                self.model = ISNet(mode='test')
            self.cal_loss = ISNetLoss()
        elif model_name == 'RISTDnet':
            self.model = RISTDnet()
        elif model_name == 'UIUNet':
            if mode == 'train':
                self.model = UIUNet(mode='train')
            else:
                self.model = UIUNet(mode='test')
        elif model_name == 'U-Net':
            self.model = Unet()
        elif model_name == 'ISTDU-Net':
            self.model = ISTDU_Net()
        elif model_name == 'RDIAN':
            self.model = RDIAN()



        elif model_name == 'DexiNed':
            self.model = DexiNed1()
        elif model_name == 'DexiNed0':
            self.model = DexiNed0()
        elif model_name == 'DexiNed1':
            self.model = DexiNed11()
        elif model_name == 'DexiNed2':
            self.model = DexiNed2()
        elif model_name == 'DexiNed0121':
            self.model = DexiNed0121()


        elif model_name == 'GCLNet':
            self.model = GCLNet()
        elif model_name == 'UnetC4F11A23':
            self.model = UnetC4F11A23()
        elif model_name == 'UnetC4F11A23AA':
            self.model = UnetC4F11A23AA()
        elif model_name == 'UnetC4F11A23AA_plot':
            self.model = UnetC4F11A23AA_plot()
        elif model_name == 'UnetC4F11A25AA':
            self.model = UnetC4F11A25AA()
        elif model_name == 'GCLNet11111':
            self.model = GCLNet11111()

        elif model_name == 'deeplab':
            self.model = DeepLabV3()
        elif model_name == 'BiSeNet':
            self.model = BiSeNet(1, 'resnet18')
        elif model_name == 'ABC':
            self.model = ABC()
        elif model_name == 'SCTransNet':
            self.model = SCTransNet()
        elif model_name == 'HCFNet':
            self.model = HCFNet()
        elif model_name == 'AGPCNet':
            self.model = AGPCNet()
        elif model_name == 'MTUNet':
            self.model = MTUNet()

        elif model_name == 'GCLNet0':
            self.model = GCLNet0()
        elif model_name == 'GCLNet1':
            self.model = GCLNet1()
        elif model_name == 'GCLNet2':
            self.model = GCLNet2()
        elif model_name == 'GCLNet3':
            self.model = GCLNet3()
        elif model_name == 'GCLNet12':
            self.model = GCLNet12()
        elif model_name == 'pt2':
            self.model = pt2()
        elif model_name == 'pt4':
            self.model = pt4()
        elif model_name == 'pt16':
            self.model = pt16()
        elif model_name == 'LCL1':
            self.model = LCL1()
        elif model_name == 'LCL2':
            self.model = LCL2()
        elif model_name == 'LCL0':
            self.model = LCL0()
        elif model_name == 'ResNet50':
            self.model = ResNet50()
        elif model_name == 'ResNet18':
            self.model = ResNet18()
        elif model_name == 'ResNet34':
            self.model = ResNet34()
        
    def forward(self, img, imgname=None):
        if imgname:
            return self.model(img, imgname)
        elif imgname is None:
            return self.model(img)



    def loss(self, pred, gt_mask):
        loss = self.cal_loss(pred, gt_mask)
        return loss

    def loss_focal(self, pred, gt_mask):
        loss = self.focal_loss.forward(pred, gt_mask)
        return loss
    def loss_tversky(self, pred, gt_mask):
        loss = self.turskey_loss(pred, gt_mask)
        return loss

    def loss_corss(self, pred, gt_mask):
        a = 0.6
        b = 1 - a
        loss = a * self.focal_loss.forward(pred, gt_mask) + b * self.cal_loss(pred, gt_mask)

        return loss

