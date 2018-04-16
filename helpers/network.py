import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

class myNet(nn.Module):#simple 3 layer unary network to output
    def __init__(self,isTest_):
        super(myNet,self).__init__()
        self.inFeat = np.int(600)
        self.isTest = isTest_
        self.layer1 = nn.Sequential(nn.BatchNorm1d(self.inFeat),
            nn.Linear(self.inFeat,1024),
            nn.ReLU())
        self.layer2 = nn.Sequential(nn.BatchNorm1d(1024),
            nn.Linear(1024,512),
            nn.ReLU())
        self.layer3 = nn.Sequential(nn.BatchNorm1d(512),
            nn.Linear(512,512),
            nn.ReLU())
        self.layer4 = nn.Sequential(nn.BatchNorm1d(512),
            nn.Linear(512,2))
    def forward(self,x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        if self.isTest: x=F.softmax(x)
        return x


