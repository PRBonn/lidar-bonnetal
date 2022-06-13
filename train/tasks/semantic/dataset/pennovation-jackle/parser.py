import os
import numpy as np
import torch
from torch.utils.data import Dataset
# from common.laserscan import LaserScan, SemLaserScan

EXTENSIONS_SCAN = ['.pcd']
EXTENSIONS_LABEL = ['.pcd']
# import time

import open3d as o3d
# from torch import clamp
from pypcd import pypcd
import re

def is_scan(filename):
  return any(filename.endswith(ext) for ext in EXTENSIONS_SCAN)


def is_label(filename):
  return any(filename.endswith(ext) for ext in EXTENSIONS_LABEL)


class SemanticKitti(Dataset):

  def __init__(self, root,    # directory where data is
               sequences,     # sequences for this data (e.g. [1,3,4,6])
               labels,        # label dict: (e.g 10: "car")
               color_map,     # colors dict bgr (e.g 10: [255, 0, 0])
               learning_map,  # classes to learn (0 to N-1 for xentropy)
               learning_map_inv,    # inverse of previous (recover labels)
               sensor,              # sensor to parse scans from
               max_points=150000,   # max number of points present in dataset
               gt=True):            # send ground truth?
    # save deats
    self.root = os.path.join(root, "sequences")
    self.sequences = sequences
    self.labels = labels
    self.color_map = color_map
    self.learning_map = learning_map
    self.learning_map_inv = learning_map_inv
    self.sensor = sensor
    self.sensor_img_H = sensor["img_prop"]["height"]
    self.sensor_img_W = sensor["img_prop"]["width"]
    self.sensor_img_means = torch.tensor(sensor["img_means"],
                                         dtype=torch.float)
    self.sensor_img_stds = torch.tensor(sensor["img_stds"],
                                        dtype=torch.float)
    self.sensor_fov_up = sensor["fov_up"]
    self.sensor_fov_down = sensor["fov_down"]
    self.max_points = max_points
    self.gt = gt

    # get number of classes (can't be len(self.learning_map) because there
    # are multiple repeated entries, so the number that matters is how many
    # there are for the xentropy)
    self.nclasses = len(self.learning_map_inv)

    # sanity checks

    # make sure directory exists
    if os.path.isdir(self.root):
      print("Sequences folder exists! Using sequences from %s" % self.root)
    else:
      raise ValueError("Sequences folder doesn't exist! Exiting...")

    # make sure labels is a dict
    assert(isinstance(self.labels, dict))

    # make sure color_map is a dict
    assert(isinstance(self.color_map, dict))

    # make sure learning_map is a dict
    assert(isinstance(self.learning_map, dict))

    # make sure sequences is a list
    assert(isinstance(self.sequences, list))

    # placeholder for filenames
    self.scan_files = []
    self.label_files = []

    # fill in with names, checking that all sequences are complete
    for seq in self.sequences:
      # to string
      seq = '{0:02d}'.format(int(seq))

      print("parsing seq {}".format(seq))

      # get paths for each
      scan_path = os.path.join(self.root, seq, "point_clouds")
      label_path = os.path.join(self.root, seq, "point_clouds")

      # get files
      scan_files = [os.path.join(dp, f) for dp, dn, fn in os.walk(
          os.path.expanduser(scan_path)) for f in fn if is_scan(f)]
      label_files = [os.path.join(dp, f) for dp, dn, fn in os.walk(
          os.path.expanduser(label_path)) for f in fn if is_label(f)]

      # check all scans have labels
      if self.gt:
        assert(len(scan_files) == len(label_files))

      # extend list
      self.scan_files.extend(scan_files)
      self.label_files.extend(label_files)

    # sort for correspondance
    self.scan_files.sort()
    self.label_files.sort()

    print("Using {} scans from sequences {}".format(len(self.scan_files),
                                                    self.sequences))

  def __getitem__(self, index):
    # get item in tensor shape
    scan_file = self.scan_files[index]
    if self.gt:
      label_file = self.label_files[index]

    ####################### open and obtain scan #######################
    # check filename is string
    if not isinstance(scan_file, str):
      raise TypeError("Filename should be string type, "
                      "but was {type}".format(type=str(type(scan_file))))

    # check extension is a laserscan
    if not any(scan_file.endswith(ext) for ext in EXTENSIONS_SCAN):
      raise RuntimeError("Filename extension is not valid scan file.")

    # if all goes well, open pointcloud
    pcd = o3d.io.read_point_cloud(scan_file)
    scan_points = (np.asarray(pcd.points))
    pc = pypcd.PointCloud.from_path(scan_file)
    scan_remissions = pc.pc_data['intensity']
    
    # check scan makes sense
    if not isinstance(scan_points, np.ndarray):
      raise TypeError("Scan should be numpy array")
    # check remission makes sense
    if scan_remissions is not None and not isinstance(scan_remissions, np.ndarray):
      raise TypeError("Remissions should be numpy array")
    #####################################################################



    ##################laserscan projection stuff###########################
    # laser parameters
    fov_up = self.sensor_fov_up / 180.0 * np.pi      # field of view up in rad
    fov_down = self.sensor_fov_down / 180.0 * np.pi  # field of view down in rad
    fov = abs(fov_down) + abs(fov_up)  # get field of view total in rad

    # get depth of all points
    depth = np.linalg.norm(scan_points, 2, axis=1)
    depth[depth == 0] = 0.0000001 #Stop divide by 0

    ### thresholding by range (distance), ignore points that are far away, only consider points within the given range
    threshold_by_range = False # keep this as an option for future use
    if threshold_by_range:
      scan_mask = depth > 30.0
      # get scan components
      scan_x = scan_points[:, 0]
      scan_y = scan_points[:, 1]
      scan_z = scan_points[:, 2]
      depth[scan_mask] = 0.00000001
      scan_x[scan_mask] = 0
      scan_y[scan_mask] = 0
      scan_z[scan_mask] = 0
      scan_remissions[scan_mask] = 0
    else:
      # get scan components
      scan_x = scan_points[:, 0]
      scan_y = scan_points[:, 1]
      scan_z = scan_points[:, 2]

    yaw = -np.arctan2(scan_y, scan_x)
    pitch = np.arcsin(scan_z / depth)

    # get projections in image coords
    scan_proj_x = 0.5 * (yaw / np.pi + 1.0)          # in [0.0, 1.0]
    scan_proj_y = 1.0 - (pitch + abs(fov_down)) / fov        # in [0.0, 1.0]

    # scale to image size using angular resolution
    scan_proj_x *= self.sensor_img_W                              # in [0.0, W]
    scan_proj_y *= self.sensor_img_H                              # in [0.0, H]

    # round and clamp for use as index
    scan_proj_x = np.floor(scan_proj_x)
    scan_proj_x = np.minimum(self.sensor_img_W - 1, scan_proj_x)
    scan_proj_x[scan_proj_x < 0] = 0
    scan_proj_x = np.maximum(0, scan_proj_x).astype(np.int32)   # in [0,W-1]

    scan_proj_y = np.floor(scan_proj_y)
    scan_proj_y = np.minimum(self.sensor_img_H - 1, scan_proj_y)
    scan_proj_y[scan_proj_y < 0] = 0
    scan_proj_y = np.maximum(0, scan_proj_y).astype(np.int32)   # in [0,H-1]

    # order in decreasing depth -- removed
    indices = np.arange(depth.shape[0])

    # assing to images
    # projected range image - [H,W] range (-1 is no data)
    scan_proj_range = np.full((self.sensor_img_H, self.sensor_img_W), -1, dtype=np.float32)
    scan_proj_range[scan_proj_y, scan_proj_x] = depth
    # projected point cloud xyz - [H,W,3] xyz coord (-1 is no data)
    scan_proj_xyz = np.full((self.sensor_img_H, self.sensor_img_W, 3), -1,dtype=np.float32)
    scan_proj_xyz[scan_proj_y, scan_proj_x] = scan_points
    # projected remission - [H,W] intensity (-1 is no data)
    scan_proj_remission = np.full((self.sensor_img_H, self.sensor_img_W), -1,dtype=np.float32)
    scan_proj_remission[scan_proj_y, scan_proj_x] = scan_remissions
    # projected index (for each pixel, what I am in the pointcloud), [H,W] index (-1 is no data)
    scan_proj_idx = np.full((self.sensor_img_H, self.sensor_img_W), -1,dtype=np.int32) 
    scan_proj_idx[scan_proj_y, scan_proj_x] = indices
    scan_proj_mask = (scan_proj_idx > 0).astype(np.int32)
    #####################################################################


    if self.gt:
      ####################### open label #######################
      key = os.path.split(label_file)[-1][:-4]
      original_filename = label_file
      number = re.findall(r'[0-9]+', key)[0]
      label_file = os.path.join(os.path.sep.join(original_filename.split(os.sep)[:-2]), "labels",
                              "label_" + number + ".npy")
      if os.path.isfile(label_file):
        label = np.load(label_file)
      else:
        label_file = os.path.join(os.path.sep.join(original_filename.split(os.sep)[:-2]), "predictions",
                                key + ".npy")
        print(label_file)
        label = np.load(label_file)
      
      # set labels
      # check label makes sense
      if not isinstance(label, np.ndarray):
        raise TypeError("Label should be numpy array")
      # only fill in attribute if the right size
      if label.shape[0] == scan_points.shape[0]:
        sem_label = label.astype(int)  # semantic label in lower half
        # only map colors to labels that exist (proj_idx -1 is no data)
        mask = (scan_proj_idx >= 0)
        # projection color with semantic labels
        proj_sem_label = np.zeros((self.sensor_img_H, self.sensor_img_W), dtype=np.int32)
        proj_sem_label[mask] = sem_label[scan_proj_idx[mask]]
      else:
        raise ValueError("Scan and Label don't contain same number of points")
      

      # map unused classes to used classes
      sem_label = self.map(sem_label, self.learning_map)
      proj_sem_label = self.map(proj_sem_label, self.learning_map)
      #####################################################################





    # make a tensor of the uncompressed data (with the max num points)
    unproj_n_points = scan_points.shape[0]
    unproj_xyz = torch.full((self.max_points, 3), -1.0, dtype=torch.float)
    unproj_xyz[:unproj_n_points] = torch.from_numpy(scan_points)
    unproj_range = torch.full([self.max_points], -1.0, dtype=torch.float)
    unproj_range[:unproj_n_points] = torch.from_numpy(depth)
    unproj_remissions = torch.full([self.max_points], -1.0, dtype=torch.float)
    unproj_remissions[:unproj_n_points] = torch.from_numpy(scan_remissions)
    if self.gt:
      unproj_labels = torch.full([self.max_points], -1.0, dtype=torch.int32)
      unproj_labels[:unproj_n_points] = torch.from_numpy(sem_label)
    else:
      unproj_labels = []

    # get points and labels
    proj_range = torch.from_numpy(scan_proj_range).clone()
    proj_xyz = torch.from_numpy(scan_proj_xyz).clone()
    proj_remission = torch.from_numpy(scan_proj_remission).clone()
    proj_mask = torch.from_numpy(scan_proj_mask)
    if self.gt:
      proj_labels = torch.from_numpy(proj_sem_label).clone()
      proj_labels = proj_labels * proj_mask
    else:
      proj_labels = []
    proj_x = torch.full([self.max_points], -1, dtype=torch.long)
    proj_x[:unproj_n_points] = torch.from_numpy(scan_proj_x)
    proj_y = torch.full([self.max_points], -1, dtype=torch.long)
    proj_y[:unproj_n_points] = torch.from_numpy(scan_proj_y)
    proj = torch.cat([proj_range.unsqueeze(0).clone(),
                      proj_xyz.clone().permute(2, 0, 1),
                      proj_remission.unsqueeze(0).clone()])
    proj = (proj - self.sensor_img_means[:, None, None]
            ) / self.sensor_img_stds[:, None, None]
    proj = proj * proj_mask.float()

    # get name and sequence
    path_norm = os.path.normpath(scan_file)
    path_split = path_norm.split(os.sep)
    path_seq = path_split[-3]
    path_name = path_split[-1].replace(".bin", ".label")
    # print("path_norm: ", path_norm)
    # print("path_seq", path_seq)
    # print("path_name", path_name)

    # return
    return proj, proj_mask, proj_labels, unproj_labels, path_seq, path_name, proj_x, proj_y, proj_range, unproj_range, proj_xyz, unproj_xyz, proj_remission, unproj_remissions, unproj_n_points

  def __len__(self):
    return len(self.scan_files)

  @staticmethod
  def map(label, mapdict):
    # put label from original values to xentropy
    # or vice-versa, depending on dictionary values
    # make learning map a lookup table
    maxkey = 0
    for key, data in mapdict.items():
      if isinstance(data, list):
        nel = len(data)
      else:
        nel = 1
      if key > maxkey:
        maxkey = key
    # +100 hack making lut bigger just in case there are unknown labels
    if nel > 1:
      lut = np.zeros((maxkey + 100, nel), dtype=np.int32)
    else:
      lut = np.zeros((maxkey + 100), dtype=np.int32)
    for key, data in mapdict.items():
      try:
        lut[key] = data
      except IndexError:
        print("Wrong key ", key)
    # do the mapping
    return lut[label]


class Parser():
  # standard conv, BN, relu
  def __init__(self,
               root,              # directory for data
               train_sequences,   # sequences to train
               valid_sequences,   # sequences to validate.
               test_sequences,    # sequences to test (if none, don't get)
               labels,            # labels in data
               color_map,         # color for each label
               learning_map,      # mapping for training labels
               learning_map_inv,  # recover labels from xentropy
               sensor,            # sensor to use
               max_points,        # max points in each scan in entire dataset
               batch_size,        # batch size for train and val
               workers,           # threads to load data
               gt=True,           # get gt?
               shuffle_train=True):  # shuffle training set?
    super(Parser, self).__init__()

    # if I am training, get the dataset
    self.root = root
    self.train_sequences = train_sequences
    self.valid_sequences = valid_sequences
    self.test_sequences = test_sequences
    self.labels = labels
    self.color_map = color_map
    self.learning_map = learning_map
    self.learning_map_inv = learning_map_inv
    self.sensor = sensor
    self.max_points = max_points
    self.batch_size = batch_size
    self.workers = workers
    self.gt = gt
    self.shuffle_train = shuffle_train

    # number of classes that matters is the one for xentropy
    self.nclasses = len(self.learning_map_inv)

    # Data loading code
    self.train_dataset = SemanticKitti(root=self.root,
                                       sequences=self.train_sequences,
                                       labels=self.labels,
                                       color_map=self.color_map,
                                       learning_map=self.learning_map,
                                       learning_map_inv=self.learning_map_inv,
                                       sensor=self.sensor,
                                       max_points=max_points,
                                       gt=self.gt)

    self.trainloader = torch.utils.data.DataLoader(self.train_dataset,
                                                   batch_size=self.batch_size,
                                                   shuffle=self.shuffle_train,
                                                   num_workers=self.workers,
                                                   pin_memory=True,
                                                   drop_last=True)
    assert len(self.trainloader) > 0
    self.trainiter = iter(self.trainloader)

    self.valid_dataset = SemanticKitti(root=self.root,
                                       sequences=self.valid_sequences,
                                       labels=self.labels,
                                       color_map=self.color_map,
                                       learning_map=self.learning_map,
                                       learning_map_inv=self.learning_map_inv,
                                       sensor=self.sensor,
                                       max_points=max_points,
                                       gt=self.gt)
    print("stop")                                
    self.validloader = torch.utils.data.DataLoader(self.valid_dataset,
                                                   batch_size=self.batch_size,
                                                   shuffle=False,
                                                   num_workers=self.workers,
                                                   pin_memory=True,
                                                   drop_last=True)
    assert len(self.validloader) > 0
    self.validiter = iter(self.validloader)

    if self.test_sequences:
      self.test_dataset = SemanticKitti(root=self.root,
                                        sequences=self.test_sequences,
                                        labels=self.labels,
                                        color_map=self.color_map,
                                        learning_map=self.learning_map,
                                        learning_map_inv=self.learning_map_inv,
                                        sensor=self.sensor,
                                        max_points=max_points,
                                        gt=False)

      self.testloader = torch.utils.data.DataLoader(self.test_dataset,
                                                    batch_size=self.batch_size,
                                                    shuffle=False,
                                                    num_workers=self.workers,
                                                    pin_memory=True,
                                                    drop_last=True)
      assert len(self.testloader) > 0
      self.testiter = iter(self.testloader)

  def get_train_batch(self):
    scans = self.trainiter.next()
    return scans

  def get_train_set(self):
    return self.trainloader

  def get_valid_batch(self):
    scans = self.validiter.next()
    return scans

  def get_valid_set(self):
    return self.validloader

  def get_test_batch(self):
    scans = self.testiter.next()
    return scans

  def get_test_set(self):
    return self.testloader

  def get_train_size(self):
    return len(self.trainloader)

  def get_valid_size(self):
    return len(self.validloader)

  def get_test_size(self):
    return len(self.testloader)

  def get_n_classes(self):
    return self.nclasses

  def get_original_class_string(self, idx):
    return self.labels[idx]

  def get_xentropy_class_string(self, idx):
    return self.labels[self.learning_map_inv[idx]]

  def to_original(self, label):
    # put label in original values
    return SemanticKitti.map(label, self.learning_map_inv)

  def to_xentropy(self, label):
    # put label in xentropy values
    return SemanticKitti.map(label, self.learning_map)

  def to_color(self, label):
    # put label in original values
    label = SemanticKitti.map(label, self.learning_map_inv)
    # put label in color
    return SemanticKitti.map(label, self.color_map)
