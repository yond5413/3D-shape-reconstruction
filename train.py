import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
def evaluate_voxel_prediction(prediction, gt):
    """The prediction and gt are 3 dim voxels. Each voxel has values 1 or 0"""
    intersection = np.sum(np.logical_and(prediction, gt))
    union = np.sum(np.logical_or(prediction, gt))
    IoU = intersection / (union + 1e-6)  # Adding epsilon to avoid division by zero
    return IoU
# Custom IoU loss function
class VoxelIoULoss(nn.Module):
    def __init__(self):
        super(VoxelIoULoss, self).__init__()

    def forward(self, predicted_voxel_grid, ground_truth_voxel_grid):
        intersection = torch.sum(torch.logical_and(predicted_voxel_grid, ground_truth_voxel_grid).float())
        union = torch.sum(torch.logical_or(predicted_voxel_grid, ground_truth_voxel_grid).float())
        iou = intersection / (union + 1e-6)  # Adding epsilon to avoid division by zero
        loss = 1 - iou
        return loss

# Custom IoU accuracy metric
def calculate_voxel_iou_accuracy(predictions, targets):
    total_iou = 0
    for prediction, target in zip(predictions, targets):
        total_iou += evaluate_voxel_prediction(prediction, target)
    average_iou = total_iou / len(predictions)
    return average_iou
def loss_stetup():
    pass

def train(model,num_epochs,train_loader,val_loader,optimizer,configs):
    ### set criterion to loss
    criterion = VoxelIoULoss()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for i, data in enumerate(tqdm(train_loader, desc="Processing batches", leave=False)):#enumerate(train_loader):
            #print(f"i: {i}")#, data: {data}")
            torch.cuda.empty_cache() ## clean up gpu memory
            torch.cuda.synchronize() ## ensures that threads wait
            
            inputs, voxel_grids = data
            
            #print(f"len of inputs: {len(inputs)}, data[0]: {len(data[0])}\n {type(data[0])} {data[0].size()}")
            #print(f"data[0].size(): {data[0].size() }, data[1].size(): {data[1].size() }")
            #print(f"voxel_grid len: {len(voxel_grids)}, type: {type(voxel_grids)}, {voxel_grids.size()}")
            #print(f"voxel[0].size(): {voxel_grids[0].size() } voxel[1].size(): {voxel_grids[1].size() }")
            ######################
            inputs = inputs.to(configs.device) ### should be cuda
            voxel_grids = voxel_grids.to(configs.device)
            ######################
            optimizer.zero_grad()

            outputs = model(inputs).detach()# double check if it is usable 
            #print(f"out:{outputs.size()}")### might be the main bottleneck
            loss = criterion(outputs, voxel_grids)
            #loss.backward() NonGrad
            optimizer.step()

            running_loss += loss.item()
            if i % 100 == 99:  # Print every 100 mini-batches
                print(f'[Epoch {epoch + 1}, Batch {i + 1}] Loss: {running_loss / 100:.3f}')
                running_loss = 0.0

    # Evaluation after each epoch
    with torch.no_grad():
        model.eval()
        total_iou_accuracy = 0
        for inputs, voxel_grids in val_loader:
            outputs = model(inputs)
            predictions = (outputs > 0.5).float()  # Assuming outputs are probabilities
            total_iou_accuracy += calculate_voxel_iou_accuracy(predictions, voxel_grids)
        average_iou_accuracy = total_iou_accuracy / len(val_loader)
        print(f'Epoch {epoch + 1}, Average IoU Accuracy: {average_iou_accuracy:.3f}')
        model.train()
    print('Finished Training')