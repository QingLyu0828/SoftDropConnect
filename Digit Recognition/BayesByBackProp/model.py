import torch
import torch.nn as nn
from layers import BBB_Linear, BBB_Conv2d
from layers import ModuleWrapper

# class conv_block(nn.Module):
#     def __init__(self, in_ch, out_ch):
#         super(conv_block, self).__init__()       
#         self.conv = nn.Sequential(
#             BBB_Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
#             nn.BatchNorm2d(out_ch),
#             nn.ReLU(),
#             BBB_Conv2d(out_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=True),
#             nn.BatchNorm2d(out_ch),
#             nn.ReLU())
#         self.maxpool = nn.MaxPool2d(2, stride=2)
        
#     def forward(self, x):
#         x = self.conv(x)
#         x = self.maxpool(x)       
#         return x

# class mlp(nn.Module):
#     def __init__(self, in_ch, out_ch):
#         super(mlp, self).__init__()        
#         self.fc = nn.Sequential(
#             BBB_Linear(in_ch, 1024),
#             nn.ReLU(),
#             BBB_Linear(1024, out_ch))

#     def forward(self, x):
#         x = self.fc(x)        
#         return x


class FCNet(ModuleWrapper):

    def __init__(self):
        super(FCNet, self).__init__()      
        filters = [32, 64]
        self.priors = {
                'prior_mu': 0,
                'prior_sigma': 0.1,
                'posterior_mu_initial': (0, 0.1),  # (mean, std) normal_
                'posterior_rho_initial': (-5, 0.1),  # (mean, std) normal_
                }
        self.conv1_1 = BBB_Conv2d(1, filters[0], kernel_size=3, stride=1, padding=1, bias=True, priors=self.priors)
        self.bn1_1 = nn.BatchNorm2d(filters[0])
        self.relu1_1 = nn.ReLU()
        self.conv1_2 = BBB_Conv2d(filters[0], filters[0], kernel_size=3, stride=1, padding=1, bias=True, priors=self.priors)
        self.bn1_2 = nn.BatchNorm2d(filters[0])
        self.relu1_2 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, stride=2)
        
        self.conv2_1 = BBB_Conv2d(filters[0], filters[1], kernel_size=3, stride=1, padding=1, bias=True, priors=self.priors)
        self.bn2_1 = nn.BatchNorm2d(filters[1])
        self.relu2_1 = nn.ReLU()
        self.conv2_2 = BBB_Conv2d(filters[1], filters[1], kernel_size=3, stride=1, padding=1, bias=True, priors=self.priors)
        self.bn2_2 = nn.BatchNorm2d(filters[1])
        self.relu2_2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, stride=2)
                
        self.fc1 = BBB_Linear(filters[1]*7*7, 1024, bias=True, priors=self.priors)
        self.relu3 = nn.ReLU()
        self.fc2 = BBB_Linear(1024, 20, bias=True, priors=self.priors)
        
    def forward(self, x):
        kl = 0
        
        _kl, e1 = self.conv1_1(x)
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
        e3 = torch.flatten(e3, start_dim=1)
        _kl, e3 = self.fc1(e3)
        kl += _kl
        e3 = self.relu3(e3)
        _kl, e3 = self.fc2(e3)
        kl += _kl

        mu, sigma = e3.split(10, 1)
        
        return mu, sigma, kl
    