import sys
import numpy as np
import csv
import os
from helpers import loadCSV
import gensim.models as g
import string
import numpy as np

# CONFIG
start_alpha=0.01
infer_epoch=1000
model="enwiki_dbow/doc2vec.bin"
OUTDIR = 'feat'
train_file_ = 'data/train.csv'
test_file_ = 'data/test.csv'

#Load Training Data for feature extraction
train_data_ = loadCSV.load(train_file_,isTrain=True)
print('Loaded data train_ #records: {}'.format(len(train_data_)))
num_train_pairs = len(train_data_)

#Load Feature Extraction Model
print('Loading model {}'.format(model))
m = g.Doc2Vec.load(model)
print('Loaded model {}'.format(model))

# Pre-processing to get number of unique sentences
idx1_ = map(lambda x:x[1],train_data_)
idx2_ = map(lambda x:x[2],train_data_)
idx1_ = np.array(idx1_)
idx2_ = np.array(idx2_)
idx12_ = np.hstack([idx1_,idx2_])
idx12_ = np.unique(idx12_)
N = np.max(idx12_)+1
print('Unique sentences {}'.format(N))

#Initialize feature array feat_
feat_ = np.zeros((N,300),dtype=np.float32)
print('Initializing zero features_ feat_:{}'.format(feat_.shape))

#Feature extraction on unique training ids
processed_ = []
for i in range(num_train_pairs):
    this_pair_ = train_data_[i]
    id_1_ = this_pair_[1]
    id_2_ = this_pair_[2]
    sent_1_ = this_pair_[3].translate(None,string.punctuation).split()
    sent_2_ = this_pair_[4].translate(None,string.punctuation).split()
    if id_1_ not in processed_:
        feat_1_ = m.infer_vector(sent_1_,alpha=start_alpha,steps=infer_epoch).astype(np.float32)
        feat_[id_1_,:] = feat_1_    
        processed_.append(id_1_)
    if id_2_ not in processed_:
        feat_2_ = m.infer_vector(sent_2_,alpha=start_alpha,steps=infer_epoch).astype(np.float32)
        feat_[id_2_,:] = feat_2_
        processed_.append(id_2_)
    print('Done feature computations {}/{}'.format(i,num_train_pairs))

#Save Features to Disk
print('SAVING feature computations {}/{}'.format(i,num_train_pairs))
np.save(os.path.join(OUTDIR,'train_data_feat_.npy'),feat_)
print('SAVED feature computations {}/{}'.format(i,num_train_pairs))
