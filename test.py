import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import pyvista as pv
import os
def evaluate_voxel_prediction(prediction, gt):
    """The prediction and gt are 3 dim voxels. Each voxel has values 1 or 0"""
    intersection = torch.sum(torch.logical_and(prediction, gt).float())#np.sum(np.logical_and(prediction, gt))
    union = torch.sum(torch.logical_or(prediction, gt).float())#np.sum(np.logical_or(prediction, gt))
    IoU = intersection / (union + 1e-6)  # Adding epsilon to avoid division by zero
    return IoU
def calculate_voxel_iou_accuracy(predictions, targets):
    total_iou = 0
    for prediction, target in zip(predictions, targets):
        total_iou += evaluate_voxel_prediction(prediction, target)
    #average_iou = total_iou #/ len(predictions)
    return total_iou#average_iou
############# saem as train.py###########

#### main function for work flow
def Eval(model,test_loader,configs):
    ### TODO
    ### add test function
    ### -> have it return best predicitons
    ### plot and save best results
    num_gpus = torch.cuda.device_count()
    if num_gpus> 1:
        device_ids = list(range(num_gpus))
        model = nn.DataParallel(model,device_ids)
    model.eval()
    top5_inputs, top5_predictions, top5_ground_truths, top5_iou_scores, top5_iou_indices =test(model,test_loader,configs)
    #top5_inputs, top5_predictions, top5_ground_truths, top5_iou_scores, top5_iou_indices
    for x in range(0,5):
        i = top5_iou_indices[x]
        print(f"i: {i}")
        filename = f"best_{x}th"
        plot_and_save_top_prediction(top_prediction=top5_predictions[x],top_ground_truth=top5_ground_truths[x],file_name=filename)
    print("Images saved from best outputs goodbye ")

'''
will return best predicitons and compare with ground truth in plots
'''
def test(model,test_loader,configs):
    bar = tqdm(total=len(test_loader))
    ############################
    top5_inputs = []
    top5_predictions = []
    top5_ground_truths = []
    top5_iou_scores = []
    top5_iou_indices = []
    ############################
    with torch.no_grad():
        model.eval()
        total_iou_accuracy = 0
        for x, (inputs, voxel_grids) in enumerate(test_loader):
            inputs = inputs.to(configs.device)
            voxel_grids = voxel_grids.to(configs.device)
            outputs = model(inputs)
            predictions = outputs
            iou_scores = []
            for pred, gt in zip(predictions, voxel_grids):
                iou = evaluate_voxel_prediction(pred, gt)
                total_iou_accuracy +=iou
                iou_scores.append(iou)
            #total_iou_accuracy += calculate_voxel_iou_accuracy(predictions, voxel_grids)
            #Update the list of top 5 predictions and corresponding IoU scores
            for i, (pred, iou) in enumerate(zip(predictions, iou_scores)):
                if len(top5_iou_scores) < 5:
                    top5_inputs.append(inputs[i])
                    top5_predictions.append(pred)
                    top5_ground_truths.append(voxel_grids[i])
                    top5_iou_scores.append(iou)
                    top5_iou_indices.append(len(top5_iou_scores))
                else:
                    min_iou_index = top5_iou_scores.index(min(top5_iou_scores))
                    if iou > top5_iou_scores[min_iou_index]:
                        top5_inputs[min_iou_index] = inputs[i]
                        top5_predictions[min_iou_index] = pred
                        top5_ground_truths[min_iou_index] = voxel_grids[i]
                        top5_iou_scores[min_iou_index] = iou
                        top5_iou_indices[min_iou_index] = i
            if x%100 == 0 and x!=0:
                val = total_iou_accuracy/x
                print(f'i= {x}, Average IoU Accuracy: {val:.3f}')
            bar.update(1)
        average_iou_accuracy = total_iou_accuracy / len(test_loader)
        print(f' Average IoU Accuracy: {average_iou_accuracy:.3f}')
    return top5_inputs, top5_predictions, top5_ground_truths, top5_iou_scores, top5_iou_indices
def plot_and_save_top_prediction(top_prediction, top_ground_truth, file_name):
    # Create PyVista mesh for top prediction
    print(top_prediction.size())
    print(type(top_prediction.cpu().numpy()))
    print(type(top_prediction.cpu().numpy()).shape)
    top_pred_mesh = pv.wrap(top_prediction.cpu().numpy())
    
    # Create PyVista mesh for corresponding ground truth
    top_gt_mesh = pv.wrap(top_ground_truth.cpu().numpy())
    
    # Create a PyVista plotter
    p = pv.Plotter(notebook=False)
    
    # Add top prediction mesh to the plotter
    p.add_mesh(top_pred_mesh, color='red', opacity=0.5, lighting=True)
    
    # Add corresponding ground truth mesh to the plotter
    p.add_mesh(top_gt_mesh, color='blue', opacity=0.5, lighting=True)
    
    # Set up camera position and view
    p.camera_position = 'xy'
    p.enable_eye_dome_lighting()
    
    # Save the plot as an image
    p.show(screenshot=file_name)