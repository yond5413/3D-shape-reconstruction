import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
def evaluate_voxel_prediction(prediction, gt):
    """The prediction and gt are 3 dim voxels. Each voxel has values 1 or 0"""
    intersection = torch.sum(torch.logical_and(prediction, gt).float())#np.sum(np.logical_and(prediction, gt))
    union = torch.sum(torch.logical_or(prediction, gt).float())#np.sum(np.logical_or(prediction, gt))
    IoU = intersection / (union + 1e-6)  # Adding epsilon to avoid division by zero
    return IoU
# Custom IoU loss function
class VoxelIoULoss(nn.Module):
    def __init__(self):
        super(VoxelIoULoss, self).__init__()

    def forward(self, predicted_voxel_grid, ground_truth_voxel_grid):
        predicted_voxel_grid.requires_grad_()
        ground_truth_voxel_grid.requires_grad_()

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
def gpu_warmup(device_ids):
    print(f"GPU(s) warmups: {len(device_ids)}")
    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 30),
        nn.ReLU(),
        nn.Linear(30, 1)
    )
    # Move the model to GPUs using DataParallel
    model = nn.DataParallel(model, device_ids=device_ids)
    model.cuda()
    # Generate dummy input data
    input_data = torch.randn(1000, 10).cuda()
    # Perform a forward pass with dummy data
    model.train()
    output = model(input_data)
    # Do some additional computation
    result = torch.mean(output)  # For example, calculate the mean of the output
    # Print the result (optional)
    print("Result:", result.item())
    print("Warm-up done!!!!")
def train(model,num_epochs,train_loader,val_loader,optimizer,configs,device):
    ### set criterion to loss
    ######## wrap in dataparallel
    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        device_ids = list(range(num_gpus))
        model = nn.DataParallel(model,device_ids)
    #########################
    #criterion = nn.BCELoss()#nn.CrossEntropyLoss()#VoxelIoULoss()
    criterion = VoxelIoULoss()
    for epoch in range(num_epochs):
        if epoch ==0 and (num_gpus>0):
            gpu_warmup(device_ids)
        else:
            model.train()
            running_loss = 0.0
            progress_bar = tqdm(total=len(train_loader))
            for i, data in enumerate(train_loader):#enumerate(tqdm(train_loader, desc="Processing batches", leave=False)):#enumerate(train_loader):
                #print(f"i: {i}")#, data: {data}")
                #torch.cuda.empty_cache() ## clean up gpu memory
                #torch.cuda.synchronize() ## ensures that threads wait
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

                outputs = model(inputs)#.detach()# double check if it is usable 
                #print(f"out:{outputs.size()}")### might be the main bottleneck
                loss = criterion(outputs, voxel_grids)
                print(f"Loss: {loss}, pred:{outputs.size()}, gt :{voxel_grids.size()}")
                loss.backward()# NonGrad
                optimizer.step()
                progress_bar.update(1)
                running_loss += loss.item()
                if i % 100 == 99:  # Print every 100 mini-batches
                    print(f'[Epoch {epoch + 1}, Batch {i + 1}] Loss: {running_loss / 100:.3f}')
                    running_loss = 0.0
                # Clean up GPU memory
            del input, voxel_grids, outputs, loss
            torch.cuda.empty_cache()
            print("Validation Now!!!")
            # Evaluation after each epoch
            val_bar = tqdm(total=len(val_loader))
            with torch.no_grad():
                model.eval()
                total_iou_accuracy = 0
                for val_inputs, val_voxel_grids in val_loader:
                    val_inputs = val_inputs.to(configs.device)
                    val_voxel_grids = val_voxel_grids.to(configs.device)
                    outputs = model(val_inputs)
                    predictions = outputs
                    #predictions = (outputs > 0.5).float()  # Assuming outputs are probabilities
                    total_iou_accuracy += calculate_voxel_iou_accuracy(predictions, val_voxel_grids)
                    val_bar.update(1)
                average_iou_accuracy = total_iou_accuracy / len(val_loader)
                print(f'Epoch {epoch + 1}, Average IoU Accuracy: {average_iou_accuracy:.3f}')
                #model.train()
            if (epoch%5 ==0) or ((epoch+1)==num_epochs):
                ### save model somewhere 
                checkpoint_path = 'model.pth'
                model_module = model.module
                torch.save({
                    'model_state_dict': model_module.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch,
                    'loss': loss
                }, checkpoint_path)

    print('Finished Training')