import sys,os
import argparse
import torch
import datetime

class Config:
    def __init__(self):
        self.p = self.parser()
        #print(vars(self.p))
        if self.p.gpu == '0' and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")
        if not os.path.exists(self.p.model_path):
            os.makedirs(self.p.model_path)
        self.model_name = datetime.datetime.now().strftime('/%Y-%m-%d_%H:%M')#'/2021-12-28_12:13'       
        print("Load parser successfully!")

    def parser(self):
        parser = argparse.ArgumentParser(description='Parser For Arguments',formatter_class=argparse.ArgumentDefaultsHelpFormatter)

        #data path
        parser.add_argument('-image_train',dest='train_index',default='train.csv',help="File for train data index.")
        parser.add_argument('-image_test',dest="test_index",default='test.csv',help="File for test data index.")
        parser.add_argument('-image_path',dest="image_path",default='./images',help="File path for images data.")
        parser.add_argument('-model_path',dest="model_path",default='./models',help="File path for models.")
        #model parameters
        parser.add_argument('-num_workers', dest="num_workers",type=int, default=10, help='Number of processes to construct batches')
        parser.add_argument('-gpu', type=str, default='0', help='Set GPU Ids : Eg: For CPU = -1, For Single GPU = 0')
        parser.add_argument('-epoch', dest='max_epochs', type=int, default=50, help='Number of epochs')
        parser.add_argument('-lr', type=float, default=0.01, help='Starting Learning Rate')
        parser.add_argument('-batch', dest='batch_size', default=256, type=int, help='Batch size')
        #Test
        parser.add_argument('-test_label', type=str, default='0', help='Set if test = 1 or not = 0')
    
        args = parser.parse_args()

        return args

if __name__ == "__main__":
    config = Config()