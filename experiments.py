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
from shapenet_loading import ShapeNetDataset
from Shapenet_partial_loading import ShapeNetDataset_partial
########
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
##########
import math
def optimizer_selection(model, opt,lr):
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
def get_dataloaders(args):
    transform_train = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    curr_dir = os.getcwd()
    train_img_dir = curr_dir+'/'+'datasets/train_imgs'
    train_voxel_dir = curr_dir+'/'+'datasets/train_voxels'
    val_img_dir = curr_dir+'/'+'datasets/val_imgs'
    val_voxel_dir = curr_dir+'/'+'datasets/val_voxels'
    test_img_dir = curr_dir+'/'+'datasets/test_imgs'
    test_voxel_dir = curr_dir+'/'+'datasets/test'
    batch_size = args.batch_size 
    fraction = args.dataset_partition
    #TODO update transforms for each
    train_dataset = ShapeNetDataset_partial(train_img_dir, train_voxel_dir,transform=transform_train,fraction=fraction)
    test_dataset = ShapeNetDataset_partial(test_img_dir, train_voxel_dir,transform=transform_train,fraction=fraction)
    val_dataset = ShapeNetDataset_partial(val_img_dir, train_voxel_dir,transform=transform_train,fraction=fraction)
    '''
    #Total dataset with databelow or just seet fraction to one above
    train_dataset = ShapeNetDataset(train_img_dir, train_voxel_dir,transform=transform_train)
    test_dataset = ShapeNetDataset(test_img_dir, test_voxel_dir,transform=transform_train)
    val_dataset = ShapeNetDataset(val_img_dir, val_voxel_dir,transform=transform_train)
    '''
    ###########
    train_loader = torch.utils.data.DataLoader(
    train_dataset, batch_size=batch_size, shuffle=True, num_workers=args.num_workers)
    ###########
    val_loader = torch.utils.data.DataLoader(
    val_dataset, batch_size=batch_size, shuffle=True, num_workers=args.num_workers)
    ###########
    test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=batch_size, shuffle=True, num_workers=args.num_workers)
    
    total_samples = len(train_loader.dataset)
    batch_size = train_loader.batch_size
    num_mini_batches = math.ceil(total_samples / batch_size)
    print("For training")
    print("Total number of samples:", total_samples)
    print("Batch size:", batch_size)
    print("Number of mini-batches:", num_mini_batches)
    ####################################
    total_samples = len(val_loader.dataset)
    batch_size = train_loader.batch_size
    num_mini_batches = math.ceil(total_samples / batch_size)
    print("\n For validation")
    print("Total number of samples:", total_samples)
    print("Batch size:", batch_size)
    print("Number of mini-batches:", num_mini_batches)
    ####################################
    total_samples = len(test_loader.dataset)
    batch_size = train_loader.batch_size
    num_mini_batches = math.ceil(total_samples / batch_size)
    print("\n For testing")
    print("Total number of samples:", total_samples)
    print("Batch size:", batch_size)
    print("Number of mini-batches:", num_mini_batches)
    
    return train_loader,val_loader,test_loader
#########
## rn batch_size: 28 with 4 gpus using data parallel seems fine
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='3D reconstruction')
    parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('--device', default='cuda',type = str, help =  "device")
    parser.add_argument('--num_workers',default= 2, type= int, help = "dataloader workers")
    parser.add_argument('--data_path',default="./datasets", type= str, help = "data path")
    parser.add_argument('--opt', default ='adam',type = str ,help = "optimzer")
    parser.add_argument('--new_model', default =True,type = bool ,help = "new_model" )
    parser.add_argument('--latent_dim', default =100,type = int ,help = "new_model" )
    parser.add_argument('--epochs', default =50,type = int ,help = "number of epochs" )
    parser.add_argument('--batch_size', default =8,type = int ,help = "number of epochs" )
    parser.add_argument('--dataset_partition', default =(1/12),type = float ,help = "amount of training data used")
    args = parser.parse_args()
    
    device = args.device
    if device == "cuda":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.new_model:
        model = Autoencoder(latent_dim=args.latent_dim)
    else:
        #### load from 'model.pth'
        path = 'model.pth'
        model = Autoencoder(latent_dim=args.latent_dim)
        model.load_state_dict(torch.load(path))
        pass
    model.to(device)
    opt = optimizer_selection(model,args.opt,args.lr)
    ##################################
    print('==> Preparing data..')
    train_loader,val_loader,test_loader = get_dataloaders(args)
    print("Beginning Trainning")
    train(model=model,num_epochs=args.epochs,train_loader=train_loader,val_loader=val_loader,optimizer=opt,configs= args,device = device)
    