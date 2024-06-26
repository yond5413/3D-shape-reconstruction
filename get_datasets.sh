#!/bin/bash

##ShapeNet_URL = https://shapenet.cs.stanford.edu/iccv17/recon3d
wget --no-check-certificate "https://shapenet.cs.stanford.edu/iccv17/recon3d/train_imgs.zip"
wget --no-check-certificate "https://shapenet.cs.stanford.edu/iccv17/recon3d/train_voxels.zip"
wget --no-check-certificate "https://shapenet.cs.stanford.edu/iccv17/recon3d/test_imgs.zip"
wget --no-check-certificate "https://shapenet.cs.stanford.edu/iccv17/recon3d/test_voxels.zip"
wget --no-check-certificate "https://shapenet.cs.stanford.edu/iccv17/recon3d/val_imgs.zip"
wget --no-check-certificate "https://shapenet.cs.stanford.edu/iccv17/recon3d/val_voxels.zip"
#wget --no-check-certificate $ShapeNet_URL"/train_imgs.zip"
#wget --no-check-certificate $ShapeNet_URL"/train_voxels.zip"
#wget --no-check-certificate $ShapeNet_URL"/test_imgs.zip"
#wget --no-check-certificate $ShapeNet_URL"/test_voxels.zip"
#wget --no-check-certificate $ShapeNet_URL"/val_imgs.zip"
#wget --no-check-certificate $ShapeNet_URL"/val_voxels.zip"

echo "Downloaded zip files from Shapenet site..."
mkdir datasets
#DESTINATION_DIR = "$PWD/datasets"

unzip train_imgs.zip
unzip train_voxels.zip
unzip val_imgs.zip
unzip val_voxels.zip
unzip test_imgs.zip
unzip test_voxels.zip

mv train_imgs datasets #"$DESTINATION_DIR"
mv train_voxels datasets #"$DESTINATION_DIR"
mv val_imgs datasets #"$DESTINATION_DIR"
mv val_voxels datasets #"$DESTINATION_DIR"
mv test_imgs datasets #"$DESTINATION_DIR"
mv test datasets #"$DESTINATION_DIR"

echo "unzipped and sent data to new directory: datasets"

rm -rf train_imgs.zip
rm -rf train_voxels.zip
rm -rf val_imgs.zip
rm -rf val_voxels.zip
rm -rf test_imgs.zip
rm -rf test_voxels.zip

echo "Cleaning up"