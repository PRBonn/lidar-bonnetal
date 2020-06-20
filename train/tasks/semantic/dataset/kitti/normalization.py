#!/usr/bin/env python3
import numpy as np
from PIL import Image
import math
import argparse
import yaml
import os
from pykitti import odometry
import re
import cv2
import torch
from tqdm import tqdm

MIN_W_ANGLE = np.deg2rad(-40)
MAX_W_ANGLE = np.deg2rad(40)
MAX_POINTS = 150000

class LaserScan:
  """Class that contains LaserScan with x,y,z,r"""
  EXTENSIONS_SCAN = ['.bin']

  def __init__(self, project=False, H=64, W=1024, fov_up=3.0, fov_down=-25.0, calib_params=None, image_width=1241, image_height=376):
    self.project = project
    self.proj_H = H
    self.proj_W = W
    self.proj_fov_up = fov_up
    self.proj_fov_down = fov_down
    self.calib_params = calib_params
    self.reset()
    self.image_width = image_width
    self.image_height = image_height

  def reset(self):
    """ Reset scan members. """
    self.points = np.zeros((0, 3), dtype=np.float32)        # [m, 3]: x, y, z
    self.remissions = np.zeros((0, 1), dtype=np.float32)    # [m ,1]: remission

    # projected range image - [H,W] range (-1 is no data)
    self.proj_range = np.full((self.proj_H, self.proj_W), -1,
                              dtype=np.float32)

    # unprojected range (list of depths for each point)
    self.unproj_range = np.zeros((0, 1), dtype=np.float32)

    # projected point cloud xyz - [H,W,3] xyz coord (-1 is no data)
    self.proj_xyz = np.full((self.proj_H, self.proj_W, 3), -1,
                            dtype=np.float32)

    self.proj_rgb = np.full((self.proj_H, self.proj_W, 3), -1,
                            dtype=np.float32) # CHANGED FROM float64

    # projected remission - [H,W] intensity (-1 is no data)
    self.proj_remission = np.full((self.proj_H, self.proj_W), -1,
                                  dtype=np.float32)

    # projected index (for each pixel, what I am in the pointcloud)
    # [H,W] index (-1 is no data)
    self.proj_idx = np.full((self.proj_H, self.proj_W), -1,
                            dtype=np.int32)

    # for each point, where it is in the range image
    self.proj_x = np.zeros((0, 1), dtype=np.int32)        # [m, 1]: x
    self.proj_y = np.zeros((0, 1), dtype=np.int32)        # [m, 1]: y

    # mask containing for each pixel, if it contains a point or not
    self.proj_mask = np.zeros((self.proj_H, self.proj_W),
                              dtype=np.int32)       # [H,W] mask

  def size(self):
    """ Return the size of the point cloud. """
    return self.points.shape[0]

  def __len__(self):
    return self.size()

  def open_scan(self, filename, imagename):
    """ Open raw scan and fill in attributes
    """
    # reset just in case there was an open structure
    self.reset()
    self.filename = filename
    self.imagename = imagename

    # check filename is string
    if not isinstance(filename, str):
      raise TypeError("Filename should be string type, "
                      "but was {type}".format(type=str(type(filename))))

    # check extension is a laserscan
    if not any(filename.endswith(ext) for ext in self.EXTENSIONS_SCAN):
      raise RuntimeError("Filename extension is not valid scan file.")

    # if all goes well, open pointcloud
    scan = np.fromfile(filename, dtype=np.float32)
    scan = scan.reshape((-1, 4))

    # put in attribute
    points = scan[:, 0:3]    # get xyz
    remissions = scan[:, 3]  # get remission
    image = Image.open(imagename)
    self.set_points(points, remissions, image)

  def set_points(self, points, remissions=None, image=None):
    """ Set scan attributes (instead of opening from file)
    """
    # reset just in case there was an open structure
    self.reset()

    # check scan makes sense
    if not isinstance(points, np.ndarray):
      raise TypeError("Scan should be numpy array")

    # check remission makes sense
    if remissions is not None and not isinstance(remissions, np.ndarray):
      raise TypeError("Remissions should be numpy array")

    # put in attribute
    self.points = points    # get xyz
    if remissions is not None:
      self.remissions = remissions  # get remission
    else:
      self.remissions = np.zeros((points.shape[0]), dtype=np.float32)

    if image:
      self.image = image

    # if projection is wanted, then do it and fill in the structure
    if self.project:
      self.do_range_projection()

  def filter_points(self,points: np.array, image_width, image_height) -> np.array:
    """
    function finds indexes of points that are within image frame ( within image width and height )
    searches for
      points with x coordinate greater than zero, less than image_width
      points with y coordinate greater than zero, less than image_height
    Args:
      points: points to be filter, shape: number_points,2
      image_width: width of image frame
      image_height: height of image frame
    return:
      indexes of points that satisfy both conditions
    """
    # points with x coordinate greater than zero, less than image_width
    in_w = np.logical_and(points[:, 0] > 0, points[:, 0] < image_width)
    # points with y coordinate greater than zero, less than image_height
    in_h = np.logical_and(points[:, 1] > 0, points[:, 1] < image_height)
    return np.logical_and(in_w, in_h)

  def get_rgb(self,image_coordinates: np.array, image: np.array) -> np.array:
    """
    function gets rgb value from image

    Args:  
      images coordinates to get rgb values of
      image: rgb image
    Returns
      np array with rgb value for every point
    """

    result = np.zeros((image_coordinates.shape[0], 3), dtype=np.float32) # CHANGED FROM float64
    
    for idx in range(image_coordinates.shape[0]):
      # get pixel coordinates of point
      x, y = np.floor(image_coordinates[idx, :]).astype(np.int64)
        
      # set rgb value according to image
      result[idx, :] = np.array(image.getpixel((int(x),int(y)))).astype(np.float32) # CHANGED FROM float64

    return result

  def do_range_projection(self):
    """ Project a pointcloud into a spherical projection image.projection.
      Function takes no arguments because it can be also called externally
      if the value of the constructor was not set (in case you change your
      mind about wanting the projection)
    """
    # laser parameters
    fov_up = self.proj_fov_up / 180.0 * np.pi      # field of view up in rad
    fov_down = self.proj_fov_down / 180.0 * np.pi  # field of view down in rad
    fov = abs(fov_down) + abs(fov_up)  # get field of view total in rad


    velo_points_plane_ind = self.points[:, 0] > 0
    
    self.homogenous_points = np.concatenate(
      [self.points, np.ones((self.points.shape[0], 1))], axis=1)
    transformed_velo_to_image = np.matmul(
      self.calib_params.P_rect_20, self.calib_params.T_cam2_velo)
    transformed_velo_to_image = np.dot(
      transformed_velo_to_image, self.homogenous_points.T)
    transformed_velo_to_image = transformed_velo_to_image[:2,:] / transformed_velo_to_image[2, :]

    inds = self.filter_points(transformed_velo_to_image.T,
              image_width=self.image_width, image_height=self.image_height)

    self.image_filter_condition = np.logical_and(velo_points_plane_ind, inds)

    self.points = self.points[self.image_filter_condition,:]
    self.remissions = self.remissions[self.image_filter_condition]

    # get rgb values
    rgb = self.get_rgb(transformed_velo_to_image.T[self.image_filter_condition, :], self.image)
    self.rgb = rgb

    # get depth of all points
    depth = np.linalg.norm(self.points, 2, axis=1)
    self.depth = depth

    # get scan components
    scan_x = self.points[:, 0]
    scan_y = self.points[:, 1]
    scan_z = self.points[:, 2]

    # get angles of all points
    yaw = -np.arctan2(scan_y, scan_x)
    pitch = np.arcsin(scan_z / depth)

    # get projections in image coords
    #proj_x = 0.5 * (yaw / np.pi + 1.0)          # in [0.0, 1.0]
    proj_x = (yaw - MIN_W_ANGLE) / (MAX_W_ANGLE - MIN_W_ANGLE) # in [0.0, 1.0]
    proj_y = 1.0 - (pitch + abs(fov_down)) / fov        # in [0.0, 1.0]

    # scale to image size using angular resolution
    proj_x *= self.proj_W                              # in [0.0, W]
    proj_y *= self.proj_H                              # in [0.0, H]

    # round and clamp for use as index
    proj_x = np.floor(proj_x)
    proj_x = np.minimum(self.proj_W - 1, proj_x)
    proj_x = np.maximum(0, proj_x).astype(np.int32)   # in [0,W-1]
    self.proj_x = np.copy(proj_x)  # store a copy in orig order

    proj_y = np.floor(proj_y)
    proj_y = np.minimum(self.proj_H - 1, proj_y)
    proj_y = np.maximum(0, proj_y).astype(np.int32)   # in [0,H-1]
    self.proj_y = np.copy(proj_y)  # stope a copy in original order

    # copy of depth in original order
    self.unproj_range = np.copy(depth)

    # order in decreasing depth
    indices = np.arange(depth.shape[0])
    order = np.argsort(depth)[::-1]
    depth = depth[order]
    indices = indices[order]
    points = self.points[order]
    remission = self.remissions[order]
    rgb = rgb[order]

    proj_y = proj_y[order]
    proj_x = proj_x[order]

    # assing to images
    self.proj_range[proj_y, proj_x] = depth
    self.proj_xyz[proj_y, proj_x] = points
    self.proj_remission[proj_y, proj_x] = remission
    self.proj_idx[proj_y, proj_x] = indices
    self.proj_mask = (self.proj_idx > 0).astype(np.int32) # CHANGED FROM float32.
    self.proj_rgb[proj_y, proj_x] = rgb

EXTENSIONS_SCAN = ['.bin']
EXTENSIONS_IMAGE = ['.png']


def is_scan(filename):
  return any(filename.endswith(ext) for ext in EXTENSIONS_SCAN)

def is_image(filename):
  return any(filename.endswith(ext) for ext in EXTENSIONS_IMAGE)

if __name__== "__main__":
    parser = argparse.ArgumentParser("./normalization.py")
    parser.add_argument(
        '--dataset', '-d',
        type=str,
        required=True,
        help='Dataset to train with. No Default',
    )
    parser.add_argument(
        '--data_cfg', '-dc',
        type=str,
        required=False,
        default='../../config/labels/semantic-kitti.yaml',
        help='Classification yaml cfg file. See /config/labels for sample. No default!',
    )
    parser.add_argument(
        '--arch_cfg', '-ac',
        type=str,
        required=True,
        help='Architecture yaml cfg file. See /config/arch for sample. No default!',
    )
    FLAGS, unparsed = parser.parse_known_args()

    # open data config file
    try:
        print("Opening data config file %s" % FLAGS.data_cfg)
        DATA = yaml.safe_load(open(FLAGS.data_cfg, 'r'))
    except Exception as e:
        print(e)
        print("Error opening data yaml file.")
        quit()

    # open arch config file
    try:
        print("Opening arch config file %s" % FLAGS.arch_cfg)
        ARCH = yaml.safe_load(open(FLAGS.arch_cfg, 'r'))
    except Exception as e:
        print(e)
        print("Error opening arch yaml file.")
        quit()

    root = os.path.join(FLAGS.dataset, "sequences")
    train_sequences=DATA["split"]["train"]
    valid_sequences=DATA["split"]["valid"]
    sequences = train_sequences
    sequences.extend(valid_sequences)

    sensor = ARCH["dataset"]["sensor"]
    sensor_img_H = sensor["img_prop"]["height"]
    sensor_img_W = sensor["img_prop"]["width"]
    sensor_fov_up = sensor["fov_up"]
    sensor_fov_down = sensor["fov_down"]
    sequence_regex = re.compile('sequences\/(\d+)\/')

    total_scan_files = []
    total_image_files = []

    all_pgm = []

    # fill in with names, checking that all sequences are complete
    for seq in sequences:

        # placeholder for filenames
        scan_files = []
        image_files = []

        # to string
        seq = '{0:02d}'.format(int(seq))

        print("parsing seq {}".format(seq))

        odometry_manager = odometry(root.replace('/sequences', ''), seq)

        # get paths for each
        scan_path = os.path.join(root, seq, "velodyne")
        image_path = os.path.join(root, seq, "image_2")

        # get files
        scan_files = [os.path.join(dp, f) for dp, dn, fn in os.walk(
            os.path.expanduser(scan_path)) for f in fn if is_scan(f)]
        image_files = [os.path.join(dp, f) for dp, dn, fn in os.walk(
            os.path.expanduser(image_path)) for f in fn if is_image(f)]

        assert(len(scan_files) == len(image_files))

        # extend list
        total_scan_files.extend(scan_files)
        total_image_files.extend(image_files)

        # sort for correspondance
        scan_files.sort()
        image_files.sort()

        for scan_file, image_file in tqdm(list(zip(scan_files, image_files)), "processing seq {}".format(seq)):
            image_size = Image.open(image_file).size
            scan = LaserScan(project=True,
                    H=sensor_img_H,
                    W=sensor_img_W,
                    fov_up=sensor_fov_up,
                    fov_down=sensor_fov_down,
                    calib_params=odometry_manager.calib,
                    image_height=image_size[1],
                    image_width=image_size[0])
            
            scan.open_scan(scan_file, image_file)
            #cv2.imwrite("pgm.png", scan.proj_rgb.astype(np.uint8))
            
            # make a tensor of the uncompressed data (with the max num points)
            unproj_n_points = scan.points.shape[0]
    
            # get points and labels
            proj_range = torch.from_numpy(scan.proj_range).clone()
            proj_xyz = torch.from_numpy(scan.proj_xyz).clone()
            proj_rgb = torch.from_numpy(scan.proj_rgb).clone()
            proj_remission = torch.from_numpy(scan.proj_remission).clone()
            proj_mask = torch.from_numpy(scan.proj_mask)
           
            proj_x = torch.full([MAX_POINTS], -1, dtype=torch.long)
            proj_x[:unproj_n_points] = torch.from_numpy(scan.proj_x)
            proj_y = torch.full([MAX_POINTS], -1, dtype=torch.long)
            proj_y[:unproj_n_points] = torch.from_numpy(scan.proj_y)
            proj = torch.cat([proj_range.unsqueeze(0).clone(),
                      proj_xyz.clone().permute(2, 0, 1),
                      proj_remission.unsqueeze(0).clone()])
            proj = torch.cat((proj, proj_rgb.clone().permute(2, 0, 1)), 0)
            proj = proj * proj_mask.float()

            all_pgm.append(proj)

    stacked_pgms = torch.stack(all_pgm)
    print("Stacked Size: ", stacked_pgms.size())

    means = torch.mean(stacked_pgms, dim=[0,2,3])
    stds = torch.std(stacked_pgms, dim=[0,2,3])

    print("Means: ", means)
    print("Stds: ", stds)