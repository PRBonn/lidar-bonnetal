import os.path
import os
import open3d as o3d
import pypcd.pypcd
from pypcd import *
import numpy as np
label_path = r"\\wsl$\Ubuntu\home\tomvdon\code\test\labels\labels\\"
path = r"\\wsl$\Ubuntu\home\tomvdon\code\test\labels"
pointclouds = [file for file in os.listdir(path) if file.endswith(".pcd")]
for p in pointclouds:
    pointcloud = os.path.join(path, p)
    name = os.path.split(pointcloud)[-1][:-4]
    points = pypcd.PointCloud.from_path(pointcloud)
    label = np.asarray(points.pc_data['label'])
    np.save( label_path + name + ".npy", label)
    print("saved " + label_path + name + ".npy")

