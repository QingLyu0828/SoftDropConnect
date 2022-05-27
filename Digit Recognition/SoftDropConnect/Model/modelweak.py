import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter

use_cuda = torch.cuda.is_available()
device = torch.device("cuda:1")

def _weight_drop(module, weights, dropout):
    """
    Helper for `WeightDrop`.
    """

    for name_w in weights:
        w = getattr(module, name_w)
        del module._parameters[name_w]
        module.register_parameter(name_w + '_raw', Parameter(w))

    original_module_forward = module.forward

    def forward(*args, **kwargs):
        for name_w in weights:
            raw_w = getattr(module, name_w + '_raw')
            mask = torch.rand(raw_w.size()) < dropout
            mask = mask.to(device)           
            w = raw_w * mask * (torch.rand(raw_w.size()).to(device) * 0.5 + 0.5) + raw_w * ~mask
            w = w / (dropout*0.75 + (1-dropout))
            setattr(module, name_w, w)

        return original_module_forward(*args, **kwargs)

    setattr(module, 'forward', forward)


class WeightDropLinear(torch.nn.Linear):

    def __init__(self, *args, weight_dropout=0.5, **kwargs):
        super().__init__(*args, **kwargs)
        weights = ['weight']
        _weight_drop(self, weights, weight_dropout)


class WeightDropConv2D(torch.nn.Conv2d):

    def __init__(self, *args, weight_dropout=0.5, **kwargs):
        super().__init__(*args, **kwargs)
        weights = ['weight']
        _weight_drop(self, weights, weight_dropout)


class FCNet(nn.Module):

    def __init__(self, p=0.5):
        super(FCNet, self).__init__()
        self.p = p
        filters = [32, 64]
                
        self.conv1_1 = WeightDropConv2D(1, filters[0], kernel_size=3, stride=1, padding=1, bias=True, weight_dropout=self.p)
        self.bn1_1 = nn.BatchNorm2d(filters[0])
        self.relu1_1 = nn.ReLU()
        self.conv1_2 = WeightDropConv2D(filters[0], filters[0], kernel_size=3, stride=1, padding=1, bias=True, weight_dropout=self.p)
        self.bn1_2 = nn.BatchNorm2d(filters[0])
        self.relu1_2 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, stride=2)
        
        self.conv2_1 = WeightDropConv2D(filters[0], filters[1], kernel_size=3, stride=1, padding=1, bias=True, weight_dropout=self.p)
        self.bn2_1 = nn.BatchNorm2d(filters[1])
        self.relu2_1 = nn.ReLU()
        self.conv2_2 = WeightDropConv2D(filters[1], filters[1], kernel_size=3, stride=1, padding=1, bias=True, weight_dropout=self.p)
        self.bn2_2 = nn.BatchNorm2d(filters[1])
        self.relu2_2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, stride=2)   
        
        self.flatten = nn.Flatten()
        self.fc1 = WeightDropLinear(filters[1]*7*7, 1024, weight_dropout=self.p, bias=True)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(1024, 20, bias=True)
        
    def forward(self, x, mask=True):
        x = self.conv1_1(x)
        x = self.bn1_1(x)
        x = self.relu1_1(x)
        x = self.conv1_2(x)
        x = self.bn1_2(x)
        x = self.relu1_2(x)
        x = self.pool1(x)
        
        x = self.conv2_1(x)
        x = self.bn2_1(x)
        x = self.relu2_1(x)
        x = self.conv2_2(x)
        x = self.bn2_2(x)
        x = self.relu2_2(x)
        x = self.pool2(x)
        
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu3(x)        
        x = self.fc2(x)
        mu, sigma = x.split(10, 1)
        
        return mu, sigma
