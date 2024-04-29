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
    test(model,test_loader,configs)

##### test runs test loop
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
        for i, (inputs, voxel_grids) in test_loader:
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
            if i%100 == 0:
                val = total_iou_accuracy/i
                print(f'Average IoU Accuracy: {val:.3f}')
            bar.update(1)
        average_iou_accuracy = total_iou_accuracy / len(test_loader)
        print(f' Average IoU Accuracy: {average_iou_accuracy:.3f}')
    return top5_inputs, top5_predictions, top5_ground_truths, top5_iou_scores, top5_iou_indices