# Convert the Ian label tool's output labels into npy labels, and the range image input into pcd files
import open3d as o3d
import numpy as np
import glob
import cv2
import os
import shutil

data_dir = "/home/sam/semantic-segmentation/lidar-bonnetal/pennovation_dataset/"
fnames = glob.glob(data_dir + "labels/1*.png") # start with 1 to avoid including the viz_ stuff

save_dir_point_cloud = data_dir + "converted_scans/"
if os.path.exists(save_dir_point_cloud):
    shutil.rmtree(save_dir_point_cloud)
os.mkdir(save_dir_point_cloud)

save_dir_label = data_dir + "converted_labels/"
if os.path.exists(save_dir_label):
    shutil.rmtree(save_dir_label)
os.mkdir(save_dir_label)

for fname in fnames:
    fname_no_png = fname.split(".png")[0]
    fname_no_prefix = fname_no_png.split('/')[-1]
    scan_fname = data_dir + "scans/" + fname_no_prefix +".tiff"
    label_fname = data_dir + "labels/" + fname_no_prefix + ".png"
    print("currently loading labels and range images for scan: ", scan_fname)

    scan = cv2.imread(scan_fname, cv2.IMREAD_UNCHANGED)
    label = cv2.imread(label_fname)

    # all rays that do not have any return will be set as 0, and they are not considered during the back-propagation
    scan = np.nan_to_num(scan, copy=True, nan=0.0, posinf=None, neginf=None)
    # make sure the label for those points belong to "unlabeled" class, which is 0
    index = (scan.sum(axis=2) == 0)
    label_copy = label
    label_copy[index, :] = 0
    if (label.flatten().sum() != label_copy.flatten().sum() ):
        raise Exception("some of the rays without returns have labels, this should not happen!")
    else:
        label = label_copy

    # convert label into expected values
    # Ian's label tool convention:
    # - unlabelled: [0, 0, 0] -- expected value is 0
    # - road: [0, 0, 1]  -- expected value is 1
    # - vegetation: [0, 1, 0]  -- expected value is 2
    # - building: [1, 0, 0]  -- expected value is 3
    # - grass/sidewalk: [0, 0.4, 0]  -- expected value is 4
    # - vehicle: [0, 1, 1]  -- expected value is 5
    # - human: [1, 0, 1]  -- expected value is 6
    # - gravel: [0, 0.5, 0.5]  -- expected value is 7
    # - tree_trunk: [0.5, 0.2, 0.2]  -- expected value is 8
    # - light_pole: [1, 0.5, 0]  -- expected value is 9
    label_converted = np.zeros((label.shape[0], label.shape[1]))
    label_converted = label[:,:,0]

    # this is not needed - Ian directly export labels as 0-9
    # for i in np.arange(label.shape[0]):
    #     for j in np.arange(label.shape[1]):
    #         current_label = label[i,j,:]
    #         # - unlabelled: [0, 0, 0] -- expected value is 0
    #         if np.array_equal(current_label, np.array([0, 0, 0])):
    #             label_converted[i,j] = 0
    #         # - road: [0, 0, 1]  -- expected value is 1
    #         elif np.array_equal(current_label, np.array([0, 0, 1])):
    #             label_converted[i,j] = 1
    #         # - vegetation: [0, 1, 0]  -- expected value is 2
    #         elif np.array_equal(current_label, np.array([0, 1, 0])):
    #             label_converted[i,j] = 2
    #         # - building: [1, 0, 0]  -- expected value is 3
    #         elif np.array_equal(current_label, np.array([1, 0, 0])):
    #             label_converted[i,j] = 3
    #         # - grass/sidewalk: [0, 0.4, 0]  -- expected value is 4
    #         elif np.array_equal(current_label, np.array([0, 0.4, 0])):
    #             label_converted[i,j] = 4
    #         # - vehicle: [0, 1, 1]  -- expected value is 5
    #         elif np.array_equal(current_label, np.array([0, 1, 1])):
    #             label_converted[i,j] = 5
    #         # - human: [1, 0, 1]  -- expected value is 6
    #         elif np.array_equal(current_label, np.array([1, 0, 1])):
    #             label_converted[i,j] = 6
    #         # - gravel: [0, 0.5, 0.5]  -- expected value is 7
    #         elif np.array_equal(current_label, np.array([0, 0.5, 0.5])):
    #             label_converted[i,j] = 7
    #         # - tree_trunk: [0.5, 0.2, 0.2]  -- expected value is 8
    #         elif np.array_equal(current_label, np.array([0.5, 0.2, 0.2])):
    #             label_converted[i,j] = 8
    #         # - light_pole: [1, 0.5, 0]  -- expected value is 9
    #         elif np.array_equal(current_label, np.array([1, 0.5, 0])):
    #             label_converted[i,j] = 9
    #         else:
    #             print("Invalid label included!!")

    # convert scan image into pcd data format
    # Pass xyz to Open3D.o3d.geometry.PointCloud and visualize
    pcd = o3d.geometry.PointCloud()
    x = scan[:,:,0].flatten()
    y = scan[:,:,1].flatten()
    z = scan[:,:,2].flatten()
    xyz = np.zeros((x.shape[0],3))
    xyz[:,0] = x
    xyz[:,1] = y
    xyz[:,2] = z
    intensity = scan[:,:,3].flatten()
    intensities = np.zeros((intensity.shape[0],3))
    intensities[:,0] = intensity
    intensities[:,1] = intensity
    intensities[:,2] = intensity
    pcd.points = o3d.utility.Vector3dVector(xyz)
    # HACK: color channel will be storing intensity information
    pcd.colors = o3d.utility.Vector3dVector(intensities)

    # save point cloud as pcd files in converted_scans folder
    o3d.io.write_point_cloud(save_dir_point_cloud + "point_cloud" + "_" + str(fname_no_prefix) + ".pcd", pcd, write_ascii=True)
    print("pcds are saved in converted_scans folder!")

    # save labels as an 1-d array in converted_labels folder
    label_converted = label_converted.ravel()

    np.save(save_dir_label + "label" + "_" + str(fname_no_prefix) + ".npy", label_converted)
    print("labels are saved in converted_labels folder!")

