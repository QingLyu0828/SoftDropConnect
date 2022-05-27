from __future__ import print_function, division
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch
from Layers import BBB_Conv3d
# from Layers import ModuleWrapper


class U_Net(nn.Module):

    def __init__(self):
        super(U_Net, self).__init__()

        n1 = 16
        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.priors = {
                'prior_mu': 0,
                'prior_sigma': 0.1,
                'posterior_mu_initial': (0, 0.1),  # (mean, std) normal_
                'posterior_rho_initial': (-5, 0.1),  # (mean, std) normal_
                }
        self.conv1_1 = BBB_Conv3d(4, filters[0], kernel_size=3, stride=1, padding=1, bias=True, priors=self.priors)
        self.bn1_1 = nn.BatchNorm3d(filters[0])
        self.relu1_1 = nn.ReLU()
        self.conv1_2 = BBB_Conv3d(filters[0], filters[0], kernel_size=3, stride=1, padding=1, bias=True, priors=self.priors)
        self.bn1_2 = nn.BatchNorm3d(filters[0])
        self.relu1_2 = nn.ReLU()
        
        self.pool1 = nn.MaxPool3d(2, stride=2)
        self.conv2_1 = BBB_Conv3d(filters[0], filters[1], kernel_size=3, stride=1, padding=1, bias=True, priors=self.priors)
        self.bn2_1 = nn.BatchNorm3d(filters[1])
        self.relu2_1 = nn.ReLU()
        self.conv2_2 = BBB_Conv3d(filters[1], filters[1], kernel_size=3, stride=1, padding=1, bias=True, priors=self.priors)
        self.bn2_2 = nn.BatchNorm3d(filters[1])
        self.relu2_2 = nn.ReLU()
        
        self.pool2 = nn.MaxPool3d(2, stride=2)
        self.conv3_1 = BBB_Conv3d(filters[1], filters[2], kernel_size=3, stride=1, padding=1, bias=True, priors=self.priors)
        self.bn3_1 = nn.BatchNorm3d(filters[2])
        self.relu3_1 = nn.ReLU()
        self.conv3_2 = BBB_Conv3d(filters[2], filters[2], kernel_size=3, stride=1, padding=1, bias=True, priors=self.priors)
        self.bn3_2 = nn.BatchNorm3d(filters[2])
        self.relu3_2 = nn.ReLU()
        
        self.pool3 = nn.MaxPool3d(2, stride=2)      
        self.conv4_1 = BBB_Conv3d(filters[2], filters[3], kernel_size=3, stride=1, padding=1, bias=True, priors=self.priors)
        self.bn4_1 = nn.BatchNorm3d(filters[3])
        self.relu4_1 = nn.ReLU()
        self.conv4_2 = BBB_Conv3d(filters[3], filters[3], kernel_size=3, stride=1, padding=1, bias=True, priors=self.priors)
        self.bn4_2 = nn.BatchNorm3d(filters[3])
        self.relu4_2 = nn.ReLU()
        
        self.pool4 = nn.MaxPool3d(2, stride=2)
        self.conv5_1 = BBB_Conv3d(filters[3], filters[4], kernel_size=3, stride=1, padding=1, bias=True, priors=self.priors)
        self.bn5_1 = nn.BatchNorm3d(filters[4])
        self.relu5_1 = nn.ReLU()
        self.conv5_2 = BBB_Conv3d(filters[4], filters[4], kernel_size=3, stride=1, padding=1, bias=True, priors=self.priors)
        self.bn5_2 = nn.BatchNorm3d(filters[4])
        self.relu5_2 = nn.ReLU()

        self.up1 = nn.Upsample(scale_factor=2)
        self.conv6_1 = BBB_Conv3d(filters[4], filters[3], kernel_size=3, stride=1, padding=1, bias=True, priors=self.priors)
        self.bn6_1 = nn.BatchNorm3d(filters[3])
        self.relu6_1 = nn.ReLU()
        self.conv6_2 = BBB_Conv3d(filters[4], filters[3], kernel_size=3, stride=1, padding=1, bias=True, priors=self.priors)
        self.bn6_2 = nn.BatchNorm3d(filters[3])
        self.relu6_2 = nn.ReLU()
        self.conv6_3 = BBB_Conv3d(filters[3], filters[3], kernel_size=3, stride=1, padding=1, bias=True, priors=self.priors)
        self.bn6_3 = nn.BatchNorm3d(filters[3])
        self.relu6_3 = nn.ReLU()
        
        self.up2 = nn.Upsample(scale_factor=2)
        self.conv7_1 = BBB_Conv3d(filters[3], filters[2], kernel_size=3, stride=1, padding=1, bias=True, priors=self.priors)
        self.bn7_1 = nn.BatchNorm3d(filters[2])
        self.relu7_1 = nn.ReLU()
        self.conv7_2 = BBB_Conv3d(filters[3], filters[2], kernel_size=3, stride=1, padding=1, bias=True, priors=self.priors)
        self.bn7_2 = nn.BatchNorm3d(filters[2])
        self.relu7_2 = nn.ReLU()
        self.conv7_3 = BBB_Conv3d(filters[2], filters[2], kernel_size=3, stride=1, padding=1, bias=True, priors=self.priors)
        self.bn7_3 = nn.BatchNorm3d(filters[2])
        self.relu7_3 = nn.ReLU()        

        self.up3 = nn.Upsample(scale_factor=2)
        self.conv8_1 = BBB_Conv3d(filters[2], filters[1], kernel_size=3, stride=1, padding=1, bias=True, priors=self.priors)
        self.bn8_1 = nn.BatchNorm3d(filters[1])
        self.relu8_1 = nn.ReLU()
        self.conv8_2 = BBB_Conv3d(filters[2], filters[1], kernel_size=3, stride=1, padding=1, bias=True, priors=self.priors)
        self.bn8_2 = nn.BatchNorm3d(filters[1])
        self.relu8_2 = nn.ReLU()
        self.conv8_3 = BBB_Conv3d(filters[1], filters[1], kernel_size=3, stride=1, padding=1, bias=True, priors=self.priors)
        self.bn8_3 = nn.BatchNorm3d(filters[1])
        self.relu8_3 = nn.ReLU()

        self.up4 = nn.Upsample(scale_factor=2)
        self.conv9_1 = BBB_Conv3d(filters[1], filters[0], kernel_size=3, stride=1, padding=1, bias=True, priors=self.priors)
        self.bn9_1 = nn.BatchNorm3d(filters[0])
        self.relu9_1 = nn.ReLU()
        self.conv9_2 = BBB_Conv3d(filters[1], filters[0], kernel_size=3, stride=1, padding=1, bias=True, priors=self.priors)
        self.bn9_2 = nn.BatchNorm3d(filters[0])
        self.relu9_2 = nn.ReLU()
        self.conv9_3 = BBB_Conv3d(filters[0], filters[0], kernel_size=3, stride=1, padding=1, bias=True, priors=self.priors)
        self.bn9_3 = nn.BatchNorm3d(filters[0])
        self.relu9_3 = nn.ReLU()
        
        self.conv_out = BBB_Conv3d(filters[0], 8, kernel_size=3, stride=1, padding=1, bias=True, priors=self.priors)
        
    def forward(self, x):
        kl = 0
        # print('x', x.size())
        _kl, e1 = self.conv1_1(x)
        # print(e1.size())
        kl += _kl
        e1 = self.bn1_1(e1)
        e1 = self.relu1_1(e1)
        _kl, e1 = self.conv1_2(e1)
        kl += _kl
        e1 = self.bn1_2(e1)
        e1 = self.relu1_2(e1)

        e2 = self.pool1(e1)
        _kl, e2 = self.conv2_1(e2)
        kl += _kl
        e2 = self.bn2_1(e2)
        e2 = self.relu2_1(e2)
        _kl, e2 = self.conv2_2(e2)
        kl += _kl
        e2 = self.bn2_2(e2)
        e2 = self.relu2_2(e2)
        
        e3 = self.pool2(e2)
        _kl, e3 = self.conv3_1(e3)
        kl += _kl
        e3 = self.bn3_1(e3)
        e3 = self.relu3_1(e3)
        _kl, e3 = self.conv3_2(e3)
        kl += _kl
        e3 = self.bn3_2(e3)
        e3 = self.relu3_2(e3)

        e4 = self.pool3(e3)
        _kl, e4 = self.conv4_1(e4)
        kl += _kl
        e4 = self.bn4_1(e4)
        e4 = self.relu4_1(e4)
        _kl, e4 = self.conv4_2(e4)
        kl += _kl
        e4 = self.bn4_2(e4)
        e4 = self.relu4_2(e4)
        # print('e4', e4.size())
        
        e5 = self.pool4(e4)
        _kl, e5 = self.conv5_1(e5)
        kl += _kl
        e5 = self.bn5_1(e5)
        e5 = self.relu5_1(e5)
        _kl, e5 = self.conv5_2(e5)
        kl += _kl
        e5 = self.bn5_2(e5)
        e5 = self.relu5_2(e5)
        # print(e5.size())

        d5 = self.up1(e5)
        _kl, d5 = self.conv6_1(d5)
        kl += _kl
        d5 = self.bn6_1(d5)
        d5 = self.relu6_1(d5)
        # print('d5', d5.size())
        d4 = torch.cat((e4, d5), dim=1)
        _kl, d4 = self.conv6_2(d4)
        kl += _kl
        d4 = self.bn6_2(d4)
        d4 = self.relu6_2(d4)
        _kl, d4 = self.conv6_3(d4)
        kl += _kl
        d4 = self.bn6_3(d4)
        d4 = self.relu6_3(d4)
 
        d4 = self.up2(d4)
        _kl, d4 = self.conv7_1(d4)
        kl += _kl
        d4 = self.bn7_1(d4)
        d4 = self.relu7_1(d4)       
        d3 = torch.cat((e3, d4), dim=1)
        _kl, d3 = self.conv7_2(d3)
        kl += _kl
        d3 = self.bn7_2(d3)
        d3 = self.relu7_2(d3)
        _kl, d3 = self.conv7_3(d3)
        kl += _kl
        d3 = self.bn7_3(d3)
        d3 = self.relu7_3(d3)

        d3 = self.up3(d3)
        _kl, d3 = self.conv8_1(d3)
        kl += _kl
        d3 = self.bn8_1(d3)
        d3 = self.relu8_1(d3)       
        d2 = torch.cat((e2, d3), dim=1)
        _kl, d2= self.conv8_2(d2)
        kl += _kl
        d2 = self.bn8_2(d2)
        d2 = self.relu8_2(d2)
        _kl, d2 = self.conv8_3(d2)
        kl += _kl
        d2 = self.bn8_3(d2)
        d2 = self.relu8_3(d2)

        d2 = self.up4(d2)
        _kl, d2 = self.conv9_1(d2)
        kl += _kl
        d2 = self.bn9_1(d2)
        d2 = self.relu9_1(d2)       
        d1 = torch.cat((e1, d2), dim=1)
        _kl, d1 = self.conv9_2(d1)
        kl += _kl
        d1 = self.bn9_2(d1)
        d1 = self.relu9_2(d1)
        _kl, d1 = self.conv9_3(d1)
        kl += _kl
        d1 = self.bn9_3(d1)
        d1 = self.relu9_3(d1)

        _kl, out = self.conv_out(d1)
        kl += _kl
        
        mu, sigma = out.split(4, 1)

        return mu, sigma, kl