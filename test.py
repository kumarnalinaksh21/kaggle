import sys
import csv
import os
from helpers import loadCSV
from helpers import network
import numpy as np
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
import gensim.models as g
import codecs
import string

#CONFIG
start_alpha=0.01
infer_epoch=1000
model="enwiki_dbow/doc2vec.bin" #doc2vec feature extraction using gensim
dataset_path = 'data/test.csv'
network_path = 'model/_iter_100000.pkl'
output_path = 'test_probs.txt'

#Load Feature Extraction Model
print('Loading model {}'.format(model))
m = g.Doc2Vec.load(model)
print('Loaded model {}'.format(model))

#Load CSV
print('Loading Dataset {}'.format(dataset_path))
data_ = loadCSV.load(dataset_path,isTrain=False)
print('Loaded Dataset {}'.format(dataset_path))

#Load Network From Path
network   = network.myNet(True)
network.load_state_dict(torch.load(network_path))

OUT = open(output_path,'w')
for test_id_,test_sample_ in enumerate(data_):
    idx_ = test_sample_[0]
    st_1 = test_sample_[1].translate(None,string.punctuation).split()
    st_2 = test_sample_[2].translate(None,string.punctuation).split()
    ft_1 = m.infer_vector(st_1,alpha=start_alpha,steps=infer_epoch).astype(np.float32)
    ft_2 = m.infer_vector(st_2,alpha=start_alpha,steps=infer_epoch).astype(np.float32)
    feats = np.vstack([np.hstack([ft_1,ft_2]),np.hstack([ft_2,ft_1])])
    probs = network(Variable(torch.FloatTensor(feats)))
    prob_ = probs.data.numpy().mean(axis=0)[1]
    OUT.write('%d,%.6f\n'%(idx_,prob_))
    print('Done Testing {}/{}'.format(test_id_+1,len(data_)))
OUT.close()
     
    
