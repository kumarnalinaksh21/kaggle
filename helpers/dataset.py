import numpy as np
from helpers import loadCSV
import random
class Dataset():
    def __init__(self,batch_size_ = 100,dataset_path='data/train.csv',feature_path='feat/train_data_feat_.npy'):
        print('Loading Dataset {}'.format(dataset_path))
        data_ = loadCSV.load(dataset_path,isTrain=True)
        print('Loaded Dataset {}'.format(dataset_path))
        print('Loading Features {}'.format(feature_path))
        self.feat_ = np.load(feature_path)
        print('Loaded Features {}'.format(feature_path))
        self.inFeat = 600
        self.batch_size = batch_size_
        self.ns_ = len(data_)
        self.cur_ = 0
        self.epoch = 0
        self.idx_ = np.arange(self.ns_)
        random.shuffle(self.idx_)
        self.labels_ = np.array(map(lambda x:x[-1],data_))
        self.idx_1_  = np.array(map(lambda x:x[1],data_))
        self.idx_2_  = np.array(map(lambda x:x[2],data_))
    def next_minibatch(self):
        ids_ = self.idx_[self.cur_:min(self.cur_+self.batch_size,self.ns_)]
        self.cur_ += self.batch_size
        if self.cur_ >= self.ns_: 
            self.epoch += 1
            print('Epoch Completed {}'.format(self.epoch))
            random.shuffle(self.idx_)
            self.cur_ = 0
        labels_ = self.labels_[ids_]
        feats_1  = np.hstack([self.feat_[self.idx_1_[ids_],:],self.feat_[self.idx_2_[ids_],:]])
        feats_2  = np.hstack([self.feat_[self.idx_2_[ids_],:],self.feat_[self.idx_1_[ids_],:]])
        feats_   = np.vstack([feats_1,feats_2])
        labels_ = np.hstack([labels_,labels_])
        return feats_,labels_


