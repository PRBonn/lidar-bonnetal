import os
import numpy as np
import torch
from torch.utils.data import Dataset
from common.laserscan import LaserScan, SemLaserScan
from pykitti import odometry
import re
import PIL.Image
import pdb

EXTENSIONS_SCAN = ['.bin']
EXTENSIONS_LABEL = ['.label', '.npy', '.png']
EXTENSIONS_IMAGE = ['.png']


def is_scan(filename):
    return any(filename.endswith(ext) for ext in EXTENSIONS_SCAN)


def is_label(filename):
    return any(filename.endswith(ext) for ext in EXTENSIONS_LABEL)


def is_image(filename):
    return any(filename.endswith(ext) for ext in EXTENSIONS_IMAGE)


class SemanticKitti(Dataset):

    def __init__(self, root,  # directory where data is
                 sequences,  # sequences for this data (e.g. [1,3,4,6])
                 labels,  # label dict: (e.g 10: "car")
                 color_map,  # colors dict bgr (e.g 10: [255, 0, 0])
                 learning_map,  # classes to learn (0 to N-1 for xentropy)
                 learning_map_inv,  # inverse of previous (recover labels)
                 sensor,  # sensor to parse scans from
                 max_points=150000,  # max number of points present in dataset
                 gt=True,
                 use_rgb_image=False,  # return rgb image for model
                 rgb_image_width=None,  # resize rgb image to width, height
                 rgb_image_height=None):  # send ground truth?
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
        self.sequence_regex = re.compile('sequences\/(\d+)\/')
        self.use_rgb_image = use_rgb_image
        self.rgb_image_width = rgb_image_width
        self.rgb_image_height = rgb_image_height

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
        assert (isinstance(self.labels, dict))

        # make sure color_map is a dict
        assert (isinstance(self.color_map, dict))

        # make sure learning_map is a dict
        assert (isinstance(self.learning_map, dict))

        # make sure sequences is a list
        assert (isinstance(self.sequences, list))

        # placeholder for filenames
        self.scan_files = []
        self.label_files = []
        self.image_files = []
        self.label1_files = []
        self.label2_files = []
        self.odometry_managers = {}
        label1_files = None
        label2_files = None
        # fill in with names, checking that all sequences are complete
        for seq in self.sequences:
            # to string
            seq = '{0:02d}'.format(int(seq))

            print("parsing seq {}".format(seq))

            self.odometry_managers[seq] = odometry(self.root.replace('/sequences', ''), seq)

            # get paths for each
            scan_path = os.path.join(self.root, seq, "velodyne")
            label_path = os.path.join(self.root, seq, "labels")
            image_path = os.path.join(self.root, seq, "image_2")
            label1_path = os.path.join(self.root, seq, "deeplab_pred_segmap")
            label2_path = os.path.join(self.root, seq, "late_2")

            # get files
            scan_files = [os.path.join(dp, f) for dp, dn, fn in os.walk(
                os.path.expanduser(scan_path)) for f in fn if is_scan(f)]
            label_files = [os.path.join(dp, f) for dp, dn, fn in os.walk(
                os.path.expanduser(label_path)) for f in fn if is_label(f)]
            image_files = [os.path.join(dp, f) for dp, dn, fn in os.walk(
                os.path.expanduser(image_path)) for f in fn if is_image(f)]
            if os.path.exists(label1_path):
                label1_files = [os.path.join(dp, f) for dp, dn, fn in os.walk(
                    os.path.expanduser(label1_path)) for f in fn if is_label(f)]
            else:
                label1_files = [None] * len(scan_files)
            if os.path.exists(label2_path):
                label2_files = [os.path.join(dp, f) for dp, dn, fn in os.walk(
                    os.path.expanduser(label2_path)) for f in fn if is_label(f)]
            else:
                label2_files = [None] * len(scan_files)

            # check all scans have labels
            if self.gt:
                assert (len(scan_files) == len(label_files))
                assert (len(scan_files) == len(image_files))
                assert (len(scan_files) == len(label1_files))
                assert (len(scan_files) == len(label2_files))

            # extend list
            self.scan_files.extend(scan_files)
            self.label_files.extend(label_files)
            self.image_files.extend(image_files)
            self.label1_files.extend(label1_files)
            self.label2_files.extend(label2_files)

        # sort for correspondance
        self.scan_files.sort()
        self.label_files.sort()
        self.image_files.sort()
        if label1_files and label1_files[0] is not None:
            self.label1_files.sort()
        if label2_files and label2_files[0] is not None:
            self.label2_files.sort()

        print("Using {} scans from sequences {}".format(len(self.scan_files),
                                                        self.sequences))

    def __getitem__(self, index):
        # get item in tensor shape
        scan_file = self.scan_files[index]
        image_file = self.image_files[index]
        l1_file = self.label1_files[index]
        l2_file = self.label2_files[index]
        image_size = PIL.Image.open(image_file).size

        if self.gt:
            label_file = self.label_files[index]

        # open a semantic laserscan
        if self.gt:
            scan = SemLaserScan(self.color_map,
                                project=True,
                                H=self.sensor_img_H,
                                W=self.sensor_img_W,
                                fov_up=self.sensor_fov_up,
                                fov_down=self.sensor_fov_down,
                                calib_params=self.odometry_managers[
                                    self.sequence_regex.search(image_file).group(1)].calib,
                                image_height=image_size[1],
                                image_width=image_size[0],
                                resize_rgb_image_height=self.rgb_image_height,
                                resize_rgb_image_width=self.rgb_image_width)
        else:
            scan = LaserScan(project=True,
                             H=self.sensor_img_H,
                             W=self.sensor_img_W,
                             fov_up=self.sensor_fov_up,
                             fov_down=self.sensor_fov_down,
                             calib_params=self.odometry_managers[self.sequence_regex.search(image_file).group(1)].calib,
                             image_height=image_size[1],
                             image_width=image_size[0],
                             resize_rgb_image_height=self.rgb_image_height,
                             resize_rgb_image_width=self.rgb_image_width)

        # open and obtain scan
        scan.open_scan(scan_file, image_file, l1_file, l2_file)
        if self.gt:
            scan.open_label(label_file)
            # map unused classes to used classes (also for projection)
            scan.sem_label = self.map(scan.sem_label, self.learning_map)
            scan.proj_sem_label = self.map(scan.proj_sem_label, self.learning_map)

        # make a tensor of the uncompressed data (with the max num points)
        unproj_n_points = scan.points.shape[0]
        unproj_xyz = torch.full((self.max_points, 3), -1.0, dtype=torch.float)
        unproj_xyz[:unproj_n_points] = torch.from_numpy(scan.points)
        unproj_range = torch.full([self.max_points], -1.0, dtype=torch.float)
        unproj_range[:unproj_n_points] = torch.from_numpy(scan.unproj_range)
        unproj_remissions = torch.full([self.max_points], -1.0, dtype=torch.float)
        unproj_remissions[:unproj_n_points] = torch.from_numpy(scan.remissions)

        # print("DIMEIONSIONS")
        # print(unproj_n_points)
        # print(unproj_xyz.shape)
        # print(unproj_range)
        # print(unproj_remissions)

        if self.gt:
            unproj_labels = torch.full([self.max_points], -1.0, dtype=torch.int32)
            unproj_labels[:unproj_n_points] = torch.from_numpy(scan.sem_label)
        else:
            unproj_labels = []

        # get points and labels
        proj_range = torch.from_numpy(scan.proj_range).clone()
        proj_xyz = torch.from_numpy(scan.proj_xyz).clone()

        proj_rgb = torch.from_numpy(scan.proj_rgb).clone()
        proj_fusion_labels = torch.from_numpy(scan.proj_fusion_labels).clone()

        proj_remission = torch.from_numpy(scan.proj_remission).clone()
        proj_mask = torch.from_numpy(scan.proj_mask)
        if self.gt:
            proj_labels = torch.from_numpy(scan.proj_sem_label).clone()
            # print(proj_labels)
            # print(proj_mask)
            proj_labels = proj_labels * proj_mask
        else:
            proj_labels = []
        proj_x = torch.full([self.max_points], -1, dtype=torch.long)
        proj_x[:unproj_n_points] = torch.from_numpy(scan.proj_x)
        proj_y = torch.full([self.max_points], -1, dtype=torch.long)
        proj_y[:unproj_n_points] = torch.from_numpy(scan.proj_y)

        proj = torch.cat([proj_range.unsqueeze(0).clone(),
                          proj_xyz.clone().permute(2, 0, 1),
                          proj_remission.unsqueeze(0).clone()])

        proj = torch.cat((proj, proj_rgb.clone().permute(2, 0, 1)), 0)

        # print(proj.shape)
        # print(proj_l1l2.shape)
        proj = torch.cat((proj, proj_fusion_labels.clone().permute(2, 0, 1)), 0)
        if np.isnan(proj).any():
            raise RuntimeError("LOLOLOLO!")
        # print("AFTER CAT: ", proj, proj.shape)

        proj = (proj - self.sensor_img_means[:, None, None]
                ) / self.sensor_img_stds[:, None, None]
        proj = torch.cat((proj, proj_fusion_labels.clone().permute(2, 0, 1)), 0)
        # print("AFTER MEAN / STD: ", proj, proj.shape)

        proj = proj * proj_mask.float()

        # print("MASK: ", proj_mask, proj_mask.shape)
        # print("AFTER MASK: ", proj, proj.shape)
        # print(proj_rgb.permute(2, 0, 1).shape)

        # get name and sequence
        path_norm = os.path.normpath(scan_file)
        path_split = path_norm.split(os.sep)
        path_seq = path_split[-3]
        path_name = path_split[-1].replace(".bin", ".label")
        # print("path_norm: ", path_norm)
        # print("path_seq", path_seq)
        # print("path_name", path_name)
        image = torch.ones(())
        if self.use_rgb_image:
            image = scan.get_transformed_image()
        # return

        return proj, proj_mask, proj_labels, unproj_labels, path_seq, path_name, proj_x, proj_y, proj_range, unproj_range, proj_xyz, unproj_xyz, proj_remission, unproj_remissions, unproj_n_points, proj_rgb, proj_fusion_labels, image

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
                 root,  # directory for data
                 train_sequences,  # sequences to train
                 valid_sequences,  # sequences to validate.
                 test_sequences,  # sequences to test (if none, don't get)
                 labels,  # labels in data
                 color_map,  # color for each label
                 learning_map,  # mapping for training labels
                 learning_map_inv,  # recover labels from xentropy
                 sensor,  # sensor to use
                 max_points,  # max points in each scan in entire dataset
                 batch_size,  # batch size for train and val
                 workers,  # threads to load data
                 gt=True,  # get gt?
                 shuffle_train=True,  # shuffle training set?
                 calibration_sequences=None,
                 use_rgb_image=False,  # return rgb image for model
                 rgb_image_width=None,  # resize rgb image to width, height
                 rgb_image_height=None):
        super(Parser, self).__init__()

        # if I am training, get the dataset
        self.root = root
        self.train_sequences = train_sequences
        self.valid_sequences = valid_sequences
        self.test_sequences = test_sequences
        self.calibration_sequences = calibration_sequences
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
        self.use_rgb_image = use_rgb_image
        self.rgb_image_width = rgb_image_width  # resize rgb image to width, height
        self.rgb_image_height = rgb_image_height

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
                                           use_rgb_image=self.use_rgb_image,
                                           rgb_image_width=self.rgb_image_width,  # resize rgb image to width, height
                                           rgb_image_height=self.rgb_image_height,
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
                                           use_rgb_image=self.use_rgb_image,
                                           rgb_image_width=self.rgb_image_width,  # resize rgb image to width, height
                                           rgb_image_height=self.rgb_image_height,
                                           gt=self.gt)

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
                                              use_rgb_image=self.use_rgb_image,
                                              rgb_image_width=self.rgb_image_width,  # resize rgb image to width, height
                                              rgb_image_height=self.rgb_image_height,
                                              gt=False)

            self.testloader = torch.utils.data.DataLoader(self.test_dataset,
                                                          batch_size=self.batch_size,
                                                          shuffle=False,
                                                          num_workers=self.workers,
                                                          pin_memory=True,
                                                          drop_last=True)
            assert len(self.testloader) > 0
            self.testiter = iter(self.testloader)

            self.calibration_dataset = SemanticKitti(root=self.root,
                                                     sequences=self.calibration_sequences,
                                                     labels=self.labels,
                                                     color_map=self.color_map,
                                                     learning_map=self.learning_map,
                                                     learning_map_inv=self.learning_map_inv,
                                                     sensor=self.sensor,
                                                     max_points=max_points,
                                                     use_rgb_image=self.use_rgb_image,
                                                     rgb_image_width=self.rgb_image_width,
                                                     # resize rgb image to width, height
                                                     rgb_image_height=self.rgb_image_height,
                                                     gt=False)

            self.calibrationloader = torch.utils.data.DataLoader(self.calibration_dataset,
                                                                 batch_size=self.batch_size,
                                                                 shuffle=False,
                                                                 num_workers=self.workers,
                                                                 pin_memory=True,
                                                                 drop_last=True)
            if len(calibration_sequences) != 0:
                assert len(self.calibrationloader) > 0
            self.calibrationiter = iter(self.calibrationloader)

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

    def get_calibration_batch(self):
        scans = self.calibrationiter.next()
        return scans

    def get_calibration_set(self):
        return self.calibrationloader

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
