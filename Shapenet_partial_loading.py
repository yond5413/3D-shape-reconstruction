import os
import random
#############
import scipy.io
from PIL import Image
#################
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np
class ShapeNetDataset_partial(Dataset):
    def __init__(self, img_dir, voxel_dir, transform=None,fraction =1.0):
        self.img_dir = img_dir
        self.voxel_dir = voxel_dir
        self.transform = transform
        self.file_list = self._load_file_list()
        self.fraction = fraction
    def _load_file_list(self):
        file_list = []
        files_and_dirs = os.listdir(self.voxel_dir)
        directories = [d for d in files_and_dirs if os.path.isdir(os.path.join(self.voxel_dir, d))]
        voxel_file = 'model.mat'
        
        # Randomly shuffle the list of directories
        random.shuffle(directories)
        fraction = self.fraction
        # Determine the number of directories to include based on the fraction
        num_directories = int(len(directories) * fraction)
        
        for dir in directories[:num_directories]:
            voxel_path = os.path.join(self.voxel_dir, dir, voxel_file)
            for i in range(12):
                img_file = f"{i:03d}.png"
                img_path = os.path.join(self.img_dir, dir, img_file)
                if os.path.isfile(img_path) and os.path.isfile(voxel_path):
                    file_list.append((img_path, voxel_path))
                else:
                    if not os.path.isfile(img_path):
                        print(f"Failure: Image file {img_path} not found!")
                    if not os.path.isfile(voxel_path):
                        print(f"Failure: Voxel file {voxel_path} not found!")
                    break
        return file_list

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        #print(f"idx:{idx}, len: {self.__len__()}")
        img_path, voxel_file = self.file_list[idx]

        # Load input image
        image = Image.open(img_path).convert("RGB")
        if self.transform:
        #    for transform in self.transforms:
        #        transformed_image = transform(transformed_image)
            image = self.transform(image)
        
        # Load ground truth voxel data
        voxel_data = scipy.io.loadmat(voxel_file)['input']
        #print(f"type:{voxel_data}")
        return image, torch.from_numpy(voxel_data).float()
if __name__ == "__main__":
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
    img_array = np.array(img_sample)
    # Check shape of the image
    print(type(img_sample))
    print("Image shape:", img_array.shape)

    # Check shape of the voxel data (AutoCAD model)
    print("Voxel data shape:", voxel_sample.shape)