from ResNet import *
from config import *
from dataloader import *
from torch.utils.data import DataLoader
from torch import optim
from torch import nn
from tqdm import tqdm
import pandas as pd
import numpy as np
import json

class Runner:
    def __init__(self):
        self.config = Config()

        if self.config.p.test_label == '1':
            self.label_dic = {int(i[1]):i[0] for i in np.loadtxt('label_dic.csv',delimiter=',',dtype=str)}
            self.model = ResNet(len(self.label_dic)).to(self.config.device)
            self.model.load_state_dict(torch.load(self.config.p.model_path + self.config.model_name + '.ckpt',map_location = self.config.device))
            self.test_set = TestSet(self.config.p.test_index)
            self.test_loader = DataLoader(
                            self.test_set,batch_size = self.config.p.batch_size,num_workers = self.config.p.num_workers, shuffle = True
            )
            self.test()
        else:
            self.train_set = TrainSet(self.config.p.train_index)
            self.train_loader = DataLoader(
                            self.train_set,batch_size = self.config.p.batch_size,num_workers = self.config.p.num_workers, shuffle = True
            )
            self.label_dic = self.train_set.label_dic
            self.model = ResNet(self.train_set.label_num).to(self.config.device)
            self.optimizer = optim.SGD(self.model.parameters(),lr = self.config.p.lr)
            self.train()
    
    def train(self):
        self.crossentroyloss = nn.CrossEntropyLoss()
        process_bar = tqdm(range(self.config.p.max_epochs))
        for epoch in process_bar:
            total_loss = 0
            acc_num = 0
            for (images,label) in self.train_loader:
                images,label = images.float().to(self.config.device),label.long().to(self.config.device)
                self.optimizer.zero_grad()
                predict,right_num = self.model.forward(images,label)
                loss = self.crossentroyloss(predict,label)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
                acc_num += right_num
            process_bar.set_description('Epoch '+str(epoch)+' Average loss is '+'{:.4}'.format(total_loss / len(self.train_set))
                +' Train data acc is '+'{:.4}'.format(acc_num / len(self.train_set)))
        
        self.save_model()
    
    def save_model(self):
        torch.save(self.model.state_dict(), self.config.p.model_path + self.config.model_name + '.ckpt')
        with open(self.config.p.model_path + self.config.model_name + '.json','w') as f:
            f.write(json.dumps(vars(self.config.p),indent=4))
        f.close()
        print('Save model successfully!')

    def test(self):
        with torch.no_grad():
            test_result = []
            test_images = []
            for (index,images) in self.test_loader:
                images = images.float().to(self.config.device)
                predict = self.model.test_predict(images).cpu().tolist()
                test_images += self.test_set.test_data_index[index.cpu().numpy()].tolist()
                test_result += predict
            with open('test_result.csv','w') as f:
                for i in range(len(test_result)):
                    f.write(test_images[i]+','+self.label_dic[test_result[i]]+'\n')
        return 
if __name__ == "__main__":
    Runner()