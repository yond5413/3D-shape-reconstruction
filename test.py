import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt #
from mpl_toolkits.mplot3d import Axes3D# 
from stl import mesh#
from mpl_toolkits import mplot3d#
## delete commented ones 
import os
import open3d as o3d
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
        print(f"{filename} for gt and pred are being generated.....")
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

def create_voxel_grid(binary_tensor, voxel_size=1.0):
    #indices = torch.nonzero(binary_tensor).cpu().float()
    #print(f"type indices:{type(indices)}")
    #z_coords = indices[:, 2]
    #norm_z = (z_coords - z_coords.min()) / (z_coords.max() - z_coords.min())
    #norm_z = norm_z.cpu()
    colormap = plt.get_cmap("viridis")
    #colors = colormap(norm_z.numpy())[:, :3]
    pcd = o3d.geometry.PointCloud()
    #indices_numpy = indices.numpy()
    indices = binary_tensor.cpu().float().numpy()
    #print(f"type indices_numpu:{type(indices_numpy)}")
    print(f"shape: {indices.shape}")
    print(f"shape: {ind.shape}")
    ind = indices[0]
    pcd.points = o3d.utility.Vector3dVector(ind)
    #pcd.points = o3d.utility.Vector3dVector(indices_numpy)
    #pcd.colors = o3d.utility.Vector3dVector(colors)
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size)
    return voxel_grid

def plot_and_save_top_prediction(top_prediction, top_ground_truth, file_name):

    ### using open3d rn 
    voxel_grid1 = create_voxel_grid(top_prediction)
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(voxel_grid1)
    vis.update_geometry(voxel_grid1)
    vis.poll_events()
    vis.update_renderer()
    pred_name = "pred_"+file_name
    vis.capture_screen_image(pred_name+".png")
    vis.destroy_window()
    ################################
    # Create and save the second voxel grid visualization
    voxel_grid2 = create_voxel_grid(top_ground_truth)
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(voxel_grid2)
    vis.update_geometry(voxel_grid2)
    vis.poll_events()
    vis.update_renderer()
    gt_name = "gt_"+file_name
    vis.capture_screen_image(gt_name+".png")
    vis.destroy_window()