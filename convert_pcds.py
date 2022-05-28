# Convert the original labeled tree-only .pcd files into .npy label files 
import open3d as o3d
import numpy as np

data_dir = "/home/sam/semantic-segmentation/lidar-bonnetal/"

for i in range(0,194):
    pointcloud = data_dir + "range_images/point_cloud" + "_"+ str(i) +".pcd"
    tree_cloud = data_dir + "range_images/tree_cloud" + "_" + str(i) + ".pcd"
    print(pointcloud)
    all_points = np.asarray(o3d.io.read_point_cloud(pointcloud).points)
    tree_points = np.asarray(o3d.io.read_point_cloud(tree_cloud).points)
    label = np.ones(len(all_points), dtype=int)
    for j, point in enumerate(all_points):
        if point in tree_points:
            label[j] = 2
    print(label.shape)
    print(all_points.shape)
    np.save("range_images/labels/label" + "_" + str(i) + ".npy", label)
    label = np.load("range_images/labels/label" + "_" + str(i) + ".npy")
    print(label.shape)
    print("Finished ", i)

