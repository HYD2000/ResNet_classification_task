import cv2
import argparse
import sys,os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset

class TrainSet(Dataset):
    def __init__(self,train_index,):
        super(TrainSet,self).__init__()
        self.train_data,self.train_data_label,self.label_dic = self.load_train(train_index)
        self.train_num,self.channel,self.image_width,self.image_height = self.train_data.shape[0],self.train_data.shape[1],self.train_data.shape[2],self.train_data.shape[3]
        self.label_num = len(self.label_dic)

    def __len__(self):
        return self.train_num
    
    def __getitem__(self, index):
        return [self.train_data[index],self.train_data_label[index]]
    
    def load_train(self,train_index):
    
        train_data_index = pd.read_csv(
                        train_index,
                        sep=',',
                        header=0,
                        )
        
        label = train_data_index.values[:,1]
        train_data_path = train_data_index.values[:,0]

        #read train data images
        print("Start load train data images ...")
        train_data = []
        for path in train_data_path:
            raw_data = cv2.imread(path)
            raw_data = raw_data.reshape(raw_data.shape[2],raw_data.shape[0],raw_data.shape[1])
            train_data.append(raw_data)
        train_data = np.array(train_data)
        print("Load train data images successfully!")
        #index label
        label_dic = {}
        num = 0
        for i in label:
            if i not in label_dic.keys():
                label_dic.update({i : num})
                num += 1
        with open('label_dic.csv','w') as f:
            for key,value in label_dic.items():
                f.write(key+','+str(value)+'\n')
        f.close()
        label_index = np.array([label_dic[i] for i in label])
        return train_data,label_index,label_dic 

class TestSet(Dataset):
    def __init__(self,test_index):
        super(TestSet,self).__init__()
        self.test_data = self.load_test(test_index)
        self.test_num,self.channel,self.image_width,self.image_height = self.test_data.shape[0],self.test_data.shape[1],self.test_data.shape[2],self.test_data.shape[3]

    def __len__(self):
        return self.test_num
    
    def __getitem__(self, index):
        return [index,self.test_data[index]]
    
    def load_test(self,test_index):
    
        self.test_data_index = np.loadtxt(test_index,dtype=str,skiprows=1)

        #read train data images
        print("Start load test data images ...")
        test_data = []
        for path in self.test_data_index:
            raw_data = cv2.imread(path)
            raw_data = raw_data.reshape(raw_data.shape[2],raw_data.shape[0],raw_data.shape[1])
            test_data.append(raw_data)
        test_data = np.array(test_data)
        print("Load test data images successfully!")
        return test_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Parser For Arguments',formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-image_train',dest='train_index',default='train.csv',help="File for train data index.")
    parser.add_argument('-image_test',dest="test_index",default='test.csv',help="File for test data index.")
    parser.add_argument('-image_path',dest="image_path",default='./images',help="File path for images data.")
    
    args = parser.parse_args()
    
    #Trainset = TrainSet(args.train_index)
    testset = TestSet(args.test_index)
    print(testset.test_data_index.shape)
