import torch
import numpy as np
from torch.autograd import Variable
import pandas as pd
import matplotlib
matplotlib.use('agg') 
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
def normal_std(x):
    return x.std() * np.sqrt((len(x) - 1.)/(len(x)))

class Data_utility(object):
    # train and valid is the ratio of training set and validation set. test = 1 - train - valid
    def __init__(self, file_name, train, valid, cuda, horizon, window, steps, normalize = 2):
        self.cuda = cuda
        self.P = window
        self.h = horizon
        self.pre_length = steps
        self.standard = StandardScaler()
        
        df = pd.read_csv(file_name)
        self.columns = ['date','SP1A_DASD_RESP','SP1A_DASD_RATE','SP1B_DASD_RESP','SP1B_DASD_RATE','SP1C_DASD_RESP',
        'SP1C_DASD_RATE','SP1D_DASD_RESP','SP1D_DASD_RATE','SP1A_MEM','SP1B_MEM','SP1C_MEM','SP1D_MEM','N_TASKS',
        'TPS','SP1A_THOUT','SP1B_THOUT','SP1C_THOUT','SP1D_THOUT','SYSPLEX_MIPS','RESP_TIME']
        self.original_columns = (len(self.columns)-1)
        self.rawdat = df[['SP1A_DASD_RESP','SP1A_DASD_RATE','SP1B_DASD_RESP','SP1B_DASD_RATE','SP1C_DASD_RESP',
        'SP1C_DASD_RATE','SP1D_DASD_RESP','SP1D_DASD_RATE','SP1A_MEM','SP1B_MEM','SP1C_MEM','SP1D_MEM','N_TASKS',
        'TPS','SP1A_THOUT','SP1B_THOUT','SP1C_THOUT','SP1D_THOUT','SYSPLEX_MIPS','RESP_TIME']].values
        self.dataNums = self.rawdat.shape[0]
        # self.dat =  self.rawdat
        self.dat = np.zeros(self.rawdat.shape)
        self.n, self.m = self.dat.shape
        # self.normalize = 2
        self.scale = np.ones(self.m)
        self._normalized(normalize)
        self._split(int(train * self.n), int((train+valid) * self.n), self.n)

        self.scale = torch.from_numpy(self.scale).float()
        if self.pre_length > 1:
            tmp = self.test[1] * self.scale.expand(self.test[1].size(0),self.pre_length, self.m)
        else:
            tmp = self.test[1] * self.scale.expand(self.test[1].size(0),self.m)
            
        if self.cuda:
            self.scale = self.scale.cuda()
        self.scale = Variable(self.scale)
        
        self.rse = normal_std(tmp)
        self.rae = torch.mean(torch.abs(tmp - torch.mean(tmp)))
    
    def _normalized(self, normalize):
        #normalized by the maximum value of entire matrix.
       
        if (normalize == 0):
            self.dat = self.rawdat
            
        if (normalize == 1):
            for i in range(self.m):
                # self.scale[i] = np.max(np.abs(self.rawdat[:,i]))
                self.dat[:,i] = self.rawdat[:,i] / np.max((self.rawdat[:,i]))
            
        #normlized by the maximum value of each row(sensor).
        if (normalize == 2):
            self.standard.fit(self.rawdat)
            self.dat = self.standard.transform(self.rawdat)
            
    def _split(self, train, valid, test):
        
        train_set = range(self.P+self.h-1, train)
        valid_set = range(train, valid)
        test_set = range(valid, self.n)
        # print(len(train_set),len(valid_set),len(test_set))
        self.train = self._batchify(train_set, self.h, 'train')
        self.valid = self._batchify(valid_set, self.h, 'valid')
        self.test = self._batchify(test_set, self.h, 'test')
        # print(self.train[1].shape,self.valid[1].shape,self.test[1].shape)
        
        
    def _batchify(self, idx_set, horizon, signal_state):
        
        if self.pre_length > 1:
            n = len(idx_set)
            X = torch.zeros((n,self.P,self.m))
            Y = torch.zeros((n,self.pre_length,self.m))

            for i in range(n):
                if signal_state == 'test' and i > n-self.pre_length-self.P:
                    break
                if signal_state == 'train' and i > n-self.pre_length-self.P:
                    break
                end = idx_set[i] - self.h + 1
                start = end - self.P
                X[i,:,:] = torch.from_numpy(self.dat[start:end, :])
                Y[i,:,:] = torch.from_numpy(self.dat[end:end+self.pre_length, :])
            return [X, Y]
        else:
            n = len(idx_set)
            X = torch.zeros((n,self.P,self.m))
            Y = torch.zeros((n,self.m))

            for i in range(n):
                end = idx_set[i] - self.h + 1
                start = end - self.P
                X[i,:,:] = torch.from_numpy(self.dat[start:end, :])
                Y[i,:] = torch.from_numpy(self.dat[idx_set[i], :])
            # Y = Y.view(Y.shape[0],1,Y.shape[1])
            return [X, Y]
    def get_batches(self, inputs, targets, batch_size, shuffle=True):
        length = len(inputs)
        if shuffle:
            index = torch.randperm(length)
        else:
            index = torch.LongTensor(range(length))
        start_idx = 0
        while (start_idx < length):
            end_idx = min(length, start_idx + batch_size)
            excerpt = index[start_idx:end_idx]
            X = inputs[excerpt] 
            Y = targets[excerpt]
            if (self.cuda):
                X = X.cuda()
                Y = Y.cuda()  
            yield Variable(X), Variable(Y)
            start_idx += batch_size

class StandardScaler():
    def __init__(self):
        self.mean = 0.
        self.std = 1.
    
    def fit(self, data):
        self.mean = data.mean(0)
        self.std = data.std(0)

    def transform(self, data):
        mean = torch.from_numpy(self.mean).type_as(data).to(data.device) if torch.is_tensor(data) else self.mean
        std = torch.from_numpy(self.std).type_as(data).to(data.device) if torch.is_tensor(data) else self.std
        return (data - mean) / std

    def inverse_transform(self, data):
        mean = torch.from_numpy(self.mean).type_as(data).to(data.device) if torch.is_tensor(data) else self.mean
        std = torch.from_numpy(self.std).type_as(data).to(data.device) if torch.is_tensor(data) else self.std
        return (data * std) + mean