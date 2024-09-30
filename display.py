import numpy as np
##############################
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt #
from mpl_toolkits.mplot3d import Axes3D# 
from stl import mesh#
from mpl_toolkits import mplot3d#
'''
Separating display from test file
-> led to issues 
'''
def Display():
    data = np.load('best_results.npz')

    # Access the arrays using the keys
    top5_inputs = data['inputs']          # Shape: (5, 256, 256)
    top5_predictions = data['predictions'] # Shape: (5, 256, 256, 256)
    top5_ground_truths = data['ground_truths'] # Shape: (5, 256, 256, 256)
    top5_iou_scores = data['iou_scores']   # Shape: (5,)
    top5_iou_indices = data['iou_indices'] # Shape: (5,)
    for x in range(0,5):
        i = top5_iou_indices[x]
        print(f"i: {i}")
        filename = f"best_{x}th"
        print(f"{filename} for gt and pred are being generated.....")
        plot_and_save_top_prediction(top_prediction=top5_predictions[x],top_ground_truth=top5_ground_truths[x],file_name=filename)
    print("Images saved from best outputs goodbye ")
    

def create_voxel_grid(binary_tensor, voxel_size=1.0,file='image.png'):
    #indices = torch.nonzero(binary_tensor).cpu().float()
    binary_array = binary_tensor.cpu().numpy()
    ################################
    # Check if all elements are either 0 or 1
    #is_binary = np.all((binary_tensor == 0) | (binary_tensor == 1))
    # check if shape is (1,256,256,256) or (256,256,256)?
    unique_values = np.unique(binary_array)

    # Check if the unique values are only 0 and 1
    is_binary = np.array_equal(unique_values, [0, 1])

    print("Is the tensor binary (method 2)?", is_binary)
    ################################

    indices = np.argwhere(binary_array[0])
    
    # Normalize z coordinates if needed (commented out)
    # z_coords = indices[:, 2]
    # norm_z = (z_coords - z_coords.min()) / (z_coords.max() - z_coords.min())
    # colormap = plt.get_cmap("viridis")
    # colors = colormap(norm_z.numpy())[:, :3]
    # Create a PointCloud object
    pcd = o3d.geometry.PointCloud()
    # Convert indices to a NumPy array
    #indices_numpy = indices.numpy()
    #ind = indices_numpy[0]
    #print(f"indices shape: {indices_numpy.shape}")
    #print(f"ind shape: {ind.shape}")
    
    print(f'shape"{binary_array.shape}')
    print(f'foo shape"{binary_array[0].shape}')
    print(f'indices shape"{indices.shape}')
    print(f'ind shape"{indices[0].shape}')
    ind = indices[0]
    # Set points in the PointCloud
    pcd.points = o3d.utility.Vector3dVector(indices.astype(np.float64))
    #pcd.points = o3d.utility.Vector3dVector(indices_numpy)
    # Optionally set colors (commented out)
    # pcd.colors = o3d.utility.Vector3dVector(colors)
    
    # Create a VoxelGrid from the PointCloud
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size,)
    width = 256#320#800
    height =256#240# 600
    print(f"w: {width}, h: {height}")
    material = o3d.visualization.rendering.MaterialRecord()
    material.shader = "defaultUnlit"
    renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)#,headless=True)
    renderer.scene.add_geometry("voxel_grid", voxel_grid, material)#o3d.visualization.rendering.MaterialRecord())
    # Set camera parameters
    print('camera stuff')
    center = np.array([128, 128, 128])
    eye = center + np.array([128, 128, 128])
    up = np.array([0, 1, 0])
    renderer.scene.camera.look_at(center, eye, up)
    renderer.scene.camera.set_projection(60.0, width / height, 0.1, 5000.0)
    print('writing ')
    # Step 7: Capture the rendered image
    image = renderer.render_to_image()
    o3d.io.write_image(file, image)
    # Cleanup
    del renderer
    #return voxel_grid
    '''#indices = torch.nonzero(binary_tensor).cpu().float()
    #print(f"type indices:{type(indices)}")
    #z_coords = indices[:, 2]
    #norm_z = (z_coords - z_coords.min()) / (z_coords.max() - z_coords.min())
    #norm_z = norm_z.cpu()
    colormap = plt.get_cmap("viridis")
    #colors = colormap(norm_z.numpy())[:, :3]
    pcd = o3d.geometry.PointCloud()
    #indices_numpy = indices.numpy()
    indices = binary_tensor.cpu().float().numpy()
    ind = indices[0]
    #print(f"type indices_numpu:{type(indices_numpy)}")
    print(f"shape: {indices.shape}")
    print(f"shape: {ind.shape}")
    pcd.points = o3d.utility.Vector3dVector(ind)
    #pcd.points = o3d.utility.Vector3dVector(indices_numpy)
    #pcd.colors = o3d.utility.Vector3dVector(colors)
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size)
    return voxel_grid'''

def plot_and_save_top_prediction(top_prediction, top_ground_truth, file_name):

    ### using open3d rn 
    #voxel_grid1 = create_voxel_grid(top_prediction)
    print('getting first one')
    pred_name = "pred_"+file_name+'.png'
    gt_name = "gt_"+file_name+'.png'
    #image_path = pred_name+".png"#"voxel_grid.png"
    create_voxel_grid(binary_tensor= top_prediction,file=pred_name)
    print('done first')
    print("getting gt now")
    create_voxel_grid(binary_tensor= top_ground_truth,file=gt_name)
    print('hopefully done')
    '''vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False)
    vis.add_geometry(voxel_grid1)
    vis.update_geometry(voxel_grid1)
    vis.poll_events()
    vis.update_renderer()
    pred_name = "pred_"+file_name
    vis.capture_screen_image(pred_name+".png")
    vis.destroy_window()'''
    #width = 800
    #height = 600
    #renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)
    #renderer.scene.add_geometry("voxel_grid", voxel_grid1, o3d.visualization.rendering.MaterialRecord())
    # Set camera parameters
    #center = np.array([128, 128, 128])
    #eye = center + np.array([128, 128, 256])
    #up = np.array([0, 1, 0])
    #renderer.scene.camera.look_at(center, eye, up)
    #renderer.scene.camera.set_projection(60.0, width / height, 0.1, 5000.0)

    # Step 7: Capture the rendered image
    #image = renderer.render_to_image()

    # Step 8: Save the image
    #pred_name = "pred_"+file_name
    #image_path = pred_name+".png"#"voxel_grid.png"

    #o3d.io.write_image(image_path, image)
    #print('done first')
    ################################
    # Create and save the second voxel grid visualization
    #voxel_grid2 = create_voxel_grid(top_ground_truth)
    #print("getting gt now")

    '''vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False)
    vis.add_geometry(voxel_grid2)
    vis.update_geometry(voxel_grid2)
    vis.poll_events()
    vis.update_renderer()
    gt_name = "gt_"+file_name
    vis.capture_screen_image(gt_name+".png")
    vis.destroy_window()'''