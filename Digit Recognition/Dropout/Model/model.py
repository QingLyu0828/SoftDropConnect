import torch.nn as nn
import torch.nn.functional as F


def MC_dropout(act_vec, p=0.5, mask=True):
    return F.dropout(act_vec, p=p, training=mask, inplace=True)


class FCNet(nn.Module):

    def __init__(self, p=0.5):
        super(FCNet, self).__init__()
        self.p = p
        filters = [32, 64]
        
        self.conv1_1 = nn.Conv2d(1, filters[0], kernel_size=3, stride=1, padding=1, bias=True)
        self.bn1_1 = nn.BatchNorm2d(filters[0])
        self.relu1_1 = nn.ReLU()
        self.conv1_2 = nn.Conv2d(filters[0], filters[0], kernel_size=3, stride=1, padding=1, bias=True)
        self.bn1_2 = nn.BatchNorm2d(filters[0])
        self.relu1_2 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, stride=2)
        
        self.conv2_1 = nn.Conv2d(filters[0], filters[1], kernel_size=3, stride=1, padding=1, bias=True)
        self.bn2_1 = nn.BatchNorm2d(filters[1])
        self.relu2_1 = nn.ReLU()
        self.conv2_2 = nn.Conv2d(filters[1], filters[1], kernel_size=3, stride=1, padding=1, bias=True)
        self.bn2_2 = nn.BatchNorm2d(filters[1])
        self.relu2_2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, stride=2)   
        
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(filters[1]*7*7, 1024, bias=True)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(1024, 20, bias=True)
        
    def forward(self, x, mask=True):
        x = self.conv1_1(x)
        x = MC_dropout(x, p=self.p, mask=mask)
        x = self.bn1_1(x)
        x = self.relu1_1(x)
        x = self.conv1_2(x)
        x = MC_dropout(x, p=self.p, mask=mask)
        x = self.bn1_2(x)
        x = self.relu1_2(x)
        x = self.pool1(x)
        
        x = self.conv2_1(x)
        x = MC_dropout(x, p=self.p, mask=mask)
        x = self.bn2_1(x)
        x = self.relu2_1(x)
        x = self.conv2_2(x)
        x = MC_dropout(x, p=self.p, mask=mask)
        x = self.bn2_2(x)
        x = self.relu2_2(x)
        x = self.pool2(x)
        
        x = self.flatten(x)
        x = self.fc1(x)
        x = MC_dropout(x, p=self.p, mask=mask)
        x = self.relu3(x)        
        x = self.fc2(x)
        mu, sigma = x.split(10, 1)
        
        return mu, sigma
    