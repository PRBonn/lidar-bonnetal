import shutil
import os
import glob
import numpy as np
from sklearn.model_selection import train_test_split
import shutil
import re

clouds = glob.glob('range_images/point_cloud_*.pcd')
train, test = train_test_split(clouds, test_size=0.20, random_state=42)
for file in train:
    shutil.copy(file, "simulated_data/sequences/00")
    number = re.findall(r'[0-9]+', file)[0]
    label = os.path.join(os.path.sep.join(file.split(os.sep)[:-1]), "labels",
                            "label_" + number + ".npy")
    shutil.copy(label, "simulated_data/sequences/00/labels")
for file in test:
    shutil.copy(file, "simulated_data/sequences/01")
    number = re.findall(r'[0-9]+', file)[0]
    label = os.path.join(os.path.sep.join(file.split(os.sep)[:-1]), "labels",
                            "label_" + number + ".npy")
    shutil.copy(label, "simulated_data/sequences/01/labels")
