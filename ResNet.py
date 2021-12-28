import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l
import numpy as np

class Residual(nn.Module):  #@save
    def __init__(self, input_channels, num_channels,
                 use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels,
                               kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels,
                               kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels,
                                   kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return F.relu(Y)

class ResNet(nn.Module):
    def __init__(self,label_num=10):
        super().__init__()
        
        self.label = torch.LongTensor(list(range(0,label_num)))

        #channel:3->64,width:width/2,height:height/2
        self.b1 = nn.Sequential(nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
                   nn.BatchNorm2d(64), nn.ReLU(),
                   nn.MaxPool2d(kernel_size=3, stride=2, padding=1))

        self.b2 = nn.Sequential(*self.resnet_block(64, 64, 2, first_block=True))
        self.b3 = nn.Sequential(*self.resnet_block(64, 128, 2))
        self.b4 = nn.Sequential(*self.resnet_block(128, 256, 2))
        self.b5 = nn.Sequential(*self.resnet_block(256, 512, 2))
        
        self.net = nn.Sequential(self.b1, self.b2, self.b3, self.b4, self.b5,
                    nn.AdaptiveAvgPool2d((1,1)),
                    nn.Flatten(), nn.Linear(512, 256))
        
        self.matrix = nn.Linear(256,label_num)
        
        self.crossentroyloss = nn.CrossEntropyLoss()
        self.crossentroyloss_one = nn.CrossEntropyLoss(reduction='none')
        
    def resnet_block(self,input_channels, num_channels, num_residuals,
                 first_block=False):
        blk = []
        for i in range(num_residuals):
            if i == 0 and not first_block:
                blk.append(Residual(input_channels, num_channels,
                                    use_1x1conv=True, strides=2))
            else:
                blk.append(Residual(num_channels, num_channels))
        return blk
    
    def forward(self,input,label):
        # input.shape = (batchsize,channels,width,height), label.shape = (batchsize)
        output = self.net(input)
        prod = self.matrix(output)
        values,predict = torch.max(prod.data,1)
        #loss = self.crossentroyloss(prod,label)
        acc_num = self.predict(predict,label.data)
        return prod,acc_num
    
    def predict(self,predict,label):
        result = torch.eq(predict,label)
        #num = np.sum(result.cpu().numpy()==True)
        right_num = result.count_nonzero(0).item()
        return right_num
    
    def test_predict(self,input):
        output = self.net(input)
        prod = self.matrix(output)
        values,predict = torch.max(prod.data,1)
        return predict

if __name__ == "__main__":
    resnet = ResNet(3)
    print(resnet.forward(torch.rand(size=(256,3,224,224)),label=torch.empty(256, dtype=torch.long).random_(3)).shape)