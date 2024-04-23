import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
#### cli configs to do 
## train/test and other model configs
import os
import argparse
#######
from model import Autoencoder
from train import train
from test import test
from shapenet_loading 
########
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
def optimizer_selection(model, opt,lr ):
    opt = opt.lower()
    print(f"opt: {opt} in the selection function")
    if opt == "sgd":
        ret = optim.SGD(model.parameters(), lr=lr,
                      momentum=0.9, weight_decay=5e-4, nesterov=False)
    elif opt == "nesterov":
        ret = optim.SGD(model.parameters(), lr=lr,
                      momentum=0.9, weight_decay=5e-4,nesterov=True)
    elif opt == "adadelta":
        ret = optim.Adadelta(model.parameters(), lr=lr,
                      weight_decay=5e-4)
    elif opt == 'adagrad':
        ret = optim.Adagrad(model.parameters(), lr=lr,
                    weight_decay=5e-4)
    elif opt == 'adam':
        ret = optim.Adam(model.parameters(), lr=lr,
                      weight_decay=5e-4)
    else:
        ### default sgd case:
        ret = optim.SGD(model.parameters(), lr=lr,
                      momentum=0.9, weight_decay=5e-4)
    return ret 
########
def get_model(args):
    pass
#########
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='3D reconstruction')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--device', default='cuda',type = str, help =  "device")
    parser.add_argument('--num_workers',default= 2, type= int, help = "dataloader workers")
    parser.add_argument('--data_path',default="./datasets", type= str, help = "data path")
    parser.add_argument('--opt', default ='sgd',type = str ,help = "optimzer")
    parser.add_argument('--new_model', default =False,type = bool ,help = "new_model" )
    parser.add_argument('--latent_dim', default =100,type = int ,help = "new_model" )
    args = parser.parse_args()
    
    device = args.device
    if device == "cuda":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Autoencoder(latent_dim=args.latent_dim)
    model.to(device)
    opt = optimizer_selection(model,args.opt,args.lr)
    ##################################
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])