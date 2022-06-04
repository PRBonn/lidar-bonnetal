import shutil
import os
import glob
import numpy as np
from sklearn.model_selection import train_test_split
import shutil
import re

for_jackle = False
if for_jackle:
    dataset_name = "pennovation_dataset_jackle"
    label_prefix = "label_sweep"
else:
    dataset_name = "pennovation_dataset"
    label_prefix = "label"
cloud_dir = "/home/sam/semantic-segmentation/lidar-bonnetal/"+dataset_name+"/converted_scans/"
label_dir = "/home/sam/semantic-segmentation/lidar-bonnetal/"+dataset_name+"/converted_labels/"
save_dir = "/home/sam/semantic-segmentation/lidar-bonnetal/"+dataset_name+"/"
clouds = glob.glob(cloud_dir + 'point_cloud_*.pcd')
test_portion = 0.1
print("percentage of data splited as test set: ", test_portion)
train, test = train_test_split(clouds, test_size=test_portion, random_state=42)

output_sequence_dir = save_dir + "sequences/"
if os.path.exists(output_sequence_dir):
    print("output_sequence_dir already exists, we are going to remove it, which are: \n" + output_sequence_dir)
    input("Press Enter to continue...")
    shutil.rmtree(output_sequence_dir)
os.makedirs(output_sequence_dir + "00/labels/")
os.makedirs(output_sequence_dir + "01/labels/")
os.makedirs(output_sequence_dir + "00/point_clouds/")
os.makedirs(output_sequence_dir + "01/point_clouds/")
os.makedirs(output_sequence_dir + "02/point_clouds/")



for file in train:
    print("loading lables for file: ", file)
    number = re.findall(r'[0-9]+', file)[0]
    shutil.copy(file, save_dir + "sequences/00/point_clouds/point_cloud_" + number + ".pcd")
    label = label_dir + label_prefix + "_" + number + ".npy"
    shutil.copy(label, save_dir + "sequences/00/labels/label_" +number+".npy")
    print("successfully loaded lables for training file: ", file)
for file in test:
    print("loading lables for file: ", file)
    number = re.findall(r'[0-9]+', file)[0]
    shutil.copy(file, save_dir + "sequences/01/point_clouds/point_cloud_" + number + ".pcd")
    label = label_dir + label_prefix + "_" + number + ".npy"
    shutil.copy(label, save_dir + "sequences/01/labels/label_" +number+".npy")
    print("successfully loaded lables for validation file: ", file)
for file in test:
    # TODO: test data should probably be created separately from validation set
    print("loading lables for file: ", file)
    number = re.findall(r'[0-9]+', file)[0]
    shutil.copy(file, save_dir + "sequences/02/point_clouds/point_cloud_" + number + ".pcd")
    print("successfully loaded lables for test file: ", file)