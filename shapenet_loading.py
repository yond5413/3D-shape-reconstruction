import os
#############
import scipy.io
from PIL import Image
#################
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class ShapeNetDataset(Dataset):
    def __init__(self, img_dir, voxel_dir, transform=None):
        self.img_dir = img_dir
        self.voxel_dir = voxel_dir
        self.transform = transform
        self.file_list = self._load_file_list()
   
    def _load_file_list(self):
        print('in load_file')
        file_list = []
        for img_file in os.listdir(self.img_dir):
            img_path = os.path.join(self.img_dir, img_file)
            print(img_path)
            voxel_file = os.path.join(self.voxel_dir, img_file.replace('.png', '.mat'))
            if os.path.isfile(voxel_file):
                file_list.append((img_path, voxel_file))
        return file_list

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_path, voxel_file = self.file_list[idx]
        
        # Load input image
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        # Load ground truth voxel data
        voxel_data = scipy.io.loadmat(voxel_file)['voxel']

        return image, torch.from_numpy(voxel_data).float()

curr_dir = os.getcwd()
train_img_dir = curr_dir+'/'+'datasets/train_imgs'
train_voxel_dir = curr_dir+'/'+'datasets/train_voxels'
val_img_dir = curr_dir+'/'+'datasets/val_imgs'
val_voxel_dir = curr_dir+'/'+'datasets/val_voxels'
test_img_dir = curr_dir+'/'+'datasets/test_imgs'
test_voxel_dir = curr_dir+'/'+'datasets/test'
#transform = transforms.Compose()
# TODO transformation
train_dataset = ShapeNetDataset(train_img_dir, train_voxel_dir)#, transform=transform)
test_dataset = ShapeNetDataset(test_img_dir, test_voxel_dir)#, transform=transform)
val_dataset = ShapeNetDataset(val_img_dir, val_voxel_dir)#, transform=transform)

print('Hi just checking something ')
img_sample, voxel_sample = train_dataset[0]

# Check shape of the image
print("Image shape:", img_sample.shape)

# Check shape of the voxel data (AutoCAD model)
print("Voxel data shape:", voxel_sample.shape)