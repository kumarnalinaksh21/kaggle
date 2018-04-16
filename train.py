import sys
import csv
import os
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
from helpers import dataset
from helpers import network

#Initialize Network
network   = network.myNet(False)
#Create Dataset with a batch_size of 500
#We use sample flipping for dataset augmentation, so actual batch size = 1000
dataset   = dataset.Dataset(500)

#Create optimizer and criterion with the CrossEntropyLoss
optimizer = optim.Adam(network.parameters(),lr=1e-3)
criterion = nn.CrossEntropyLoss()

#We train for 100K iterations
iters = 100000
for iter_ in range(1,iters+1):
    losses = []
    x_,l_ = dataset.next_minibatch()
    inputv = Variable(torch.FloatTensor(x_))
    labelsv = Variable(torch.LongTensor(l_))
    output = network(inputv)
    loss = criterion(output, labelsv)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    losses.append(loss.data.sum())
    if iter_%1000 == 0: torch.save(network.state_dict(), 'model/_iter_%d.pkl'%(iter_))
    print('[%d/%d] Loss: %.3f' % (iter_+1, iters, np.mean(losses)))
