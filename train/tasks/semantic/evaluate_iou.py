#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.

import argparse
import os
import yaml
import sys
import numpy as np
import torch
import __init__ as booger
import PIL
from pykitti import odometry
import re

from tasks.semantic.modules.ioueval import iouEval
from common.laserscan import SemLaserScan

# possible splits
splits = ["train", "valid", "test"]

if __name__ == '__main__':
  parser = argparse.ArgumentParser("./evaluate_iou.py")
  parser.add_argument(
      '--dataset', '-d',
      type=str,
      required=True,
      help='Dataset dir. No Default',
  )
  parser.add_argument(
      '--predictions', '-p',
      type=str,
      required=None,
      help='Prediction dir. Same organization as dataset, but predictions in'
      'each sequences "prediction" directory. No Default. If no option is set'
      ' we look for the labels in the same directory as dataset'
  )
  parser.add_argument(
      '--split', '-s',
      type=str,
      required=False,
      choices=["train", "valid", "test"],
      default="valid",
      help='Split to evaluate on. One of ' +
      str(splits) + '. Defaults to %(default)s',
  )
  parser.add_argument(
      '--data_cfg', '-dc',
      type=str,
      required=False,
      default="config/labels/semantic-kitti.yaml",
      help='Dataset config file. Defaults to %(default)s',
  )
  parser.add_argument(
      '--arch_cfg', '-ac',
      type=str,
      required=False,
      default="config/arch/squeezesegV2_crf.yaml",
      help='Arch config file. Defaults to %(default)s',
  )
  parser.add_argument(
      '--limit', '-l',
      type=int,
      required=False,
      default=None,
      help='Limit to the first "--limit" points of each scan. Useful for'
      ' evaluating single scan from aggregated pointcloud.'
      ' Defaults to %(default)s',
  )

  FLAGS, unparsed = parser.parse_known_args()

  # fill in real predictions dir
  if FLAGS.predictions is None:
    FLAGS.predictions = FLAGS.dataset

  # print summary of what we will do
  print("*" * 80)
  print("INTERFACE:")
  print("Data: ", FLAGS.dataset)
  print("Arch: ", FLAGS.arch_cfg)
  print("Predictions: ", FLAGS.predictions)
  print("Split: ", FLAGS.split)
  print("Config: ", FLAGS.data_cfg)
  print("Limit: ", FLAGS.limit)
  print("*" * 80)

  # assert split
  assert(FLAGS.split in splits)

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

  # get number of interest classes, and the label mappings
  class_strings = DATA["labels"]
  class_remap = DATA["learning_map"]
  class_inv_remap = DATA["learning_map_inv"]
  class_ignore = DATA["learning_ignore"]
  nr_classes = len(class_inv_remap)

  # make lookup table for mapping
  maxkey = 0
  for key, data in class_remap.items():
    if key > maxkey:
      maxkey = key
  # +100 hack making lut bigger just in case there are unknown labels
  remap_lut = np.zeros((maxkey + 100), dtype=np.int32)
  for key, data in class_remap.items():
    try:
      remap_lut[key] = data
    except IndexError:
      print("Wrong key ", key)
  # print(remap_lut)

  # create evaluator
  ignore = []
  for cl, ign in class_ignore.items():
    if ign:
      x_cl = int(cl)
      ignore.append(x_cl)
      print("Ignoring xentropy class ", x_cl, " in IoU evaluation")

  # create evaluator
  device = torch.device("cpu")
  evaluator = iouEval(nr_classes, device, ignore)
  evaluator.reset()

  # get test set
  test_sequences = DATA["split"][FLAGS.split]

  # get scan paths
  scan_names = []
  for sequence in test_sequences:
    sequence = '{0:02d}'.format(int(sequence))
    scan_paths = os.path.join(FLAGS.dataset, "sequences",
                              str(sequence), "velodyne")
    # populate the scan names
    seq_scan_names = [os.path.join(dp, f) for dp, dn, fn in os.walk(
        os.path.expanduser(scan_paths)) for f in fn if ".bin" in f]
    seq_scan_names.sort()
    scan_names.extend(seq_scan_names)
  # print(scan_names)

  # get label paths
  label_names = []
  for sequence in test_sequences:
    sequence = '{0:02d}'.format(int(sequence))
    label_paths = os.path.join(FLAGS.dataset, "sequences",
                               str(sequence), "labels")
    # populate the label names
    seq_label_names = [os.path.join(dp, f) for dp, dn, fn in os.walk(
        os.path.expanduser(label_paths)) for f in fn if ".label" in f]
    seq_label_names.sort()
    label_names.extend(seq_label_names)
  # print(label_names)

  # get predictions paths
  pred_names = []
  for sequence in test_sequences:
    sequence = '{0:02d}'.format(int(sequence))
    pred_paths = os.path.join(FLAGS.predictions, "sequences",
                              sequence, "predictions")
    # populate the label names
    seq_pred_names = [os.path.join(dp, f) for dp, dn, fn in os.walk(
        os.path.expanduser(pred_paths)) for f in fn if ".label" in f]
    seq_pred_names.sort()
    pred_names.extend(seq_pred_names)
  # print(pred_names)

  sequence_regex = re.compile('sequences\/(\d+)\/')
  odometry_managers = {}

  # get image paths
  image_names = []
  for sequence in test_sequences:
    sequence = '{0:02d}'.format(int(sequence))
    image_paths = os.path.join(FLAGS.dataset, "sequences",
                              sequence, "image_2")

    odometry_managers[sequence] = odometry(FLAGS.dataset.replace('/sequences', ''), sequence)

    # populate the image names
    seq_image_names = [os.path.join(dp, f) for dp, dn, fn in os.walk(
        os.path.expanduser(image_paths)) for f in fn if ".png" in f]
    seq_image_names.sort()
    image_names.extend(seq_image_names)
  # print(image_names)

  # check that I have the same number of files
  # print("labels: ", len(label_names))
  # print("predictions: ", len(pred_names))
  l1_names = []
  for sequence in test_sequences:
    sequence = '{0:02d}'.format(int(sequence))
    l1_paths = os.path.join(FLAGS.dataset, "sequences",
                              sequence, "deeplab_pred_segmap")
    if os.path.exists(l1_paths):
    # populate the label names
      seq_l1_names = [os.path.join(dp, f) for dp, dn, fn in os.walk(
        os.path.expanduser(l1_paths)) for f in fn if ".npy" in f]
      seq_l1_names.sort()
      l1_names.extend(seq_l1_names)
    else:
      l1_names = [None] * len(image_names)
    
  l2_names = []
  for sequence in test_sequences:
    sequence = '{0:02d}'.format(int(sequence))
    l2_paths = os.path.join(FLAGS.dataset, "sequences",
                              sequence, "late_2")
    if os.path.exists(l2_paths):
      seq_l2_names = [os.path.join(dp, f) for dp, dn, fn in os.walk(
          os.path.expanduser(l2_paths)) for f in fn if ".npy" in f]
      seq_l2_names.sort()
      l2_names.extend(seq_l2_names)
    else:
      l2_names = [None] * len(image_names)

  assert(len(label_names) == len(scan_names) and
         len(label_names) == len(pred_names) and
         len(label_names) == len(image_names) and
         len(label_names) == len(l1_names) and
         len(label_names) == len(l2_names))

  print("Evaluating sequences: ")
  # open each file, get the tensor, and make the iou comparison
  for scan_file, label_file, pred_file, image_file, l1_file, l2_file in zip(scan_names, label_names, pred_names, image_names, l1_names, l2_names):
    print("evaluating label ", label_file, "with", pred_file, "with", image_file)
    # open label
    #label = SemLaserScan(project=False)

    image_size = PIL.Image.open(image_file).size
    label = SemLaserScan(DATA["color_map"],
                          project=True,
                          H=ARCH["dataset"]["sensor"]["img_prop"]["height"],
                          W=ARCH["dataset"]["sensor"]["img_prop"]["width"],
                          fov_up=ARCH["dataset"]["sensor"]["fov_up"],
                          fov_down=ARCH["dataset"]["sensor"]["fov_down"],
                          calib_params=odometry_managers[sequence_regex.search(image_file).group(1)].calib,
                          image_height=image_size[1],
                          image_width=image_size[0],
                          resize_rgb_image_width=ARCH["backbone"]["rgb_image_width"],
                          resize_rgb_image_height=ARCH["backbone"]["rgb_image_height"])

    label.open_scan(scan_file, image_file, l1_file, l2_file)
    label.open_label(label_file)
    u_label_sem = remap_lut[label.sem_label]  # remap to xentropy format
    if FLAGS.limit is not None:
      u_label_sem = u_label_sem[:FLAGS.limit]

    # open prediction
    #pred = SemLaserScan(project=False)

    pred = SemLaserScan(DATA["color_map"],
                          project=True,
                          H=ARCH["dataset"]["sensor"]["img_prop"]["height"],
                          W=ARCH["dataset"]["sensor"]["img_prop"]["width"],
                          fov_up=ARCH["dataset"]["sensor"]["fov_up"],
                          fov_down=ARCH["dataset"]["sensor"]["fov_down"],
                          calib_params=odometry_managers[sequence_regex.search(image_file).group(1)].calib,
                          image_height=image_size[1],
                          image_width=image_size[0],
                          resize_rgb_image_width=ARCH["backbone"]["rgb_image_width"],
                          resize_rgb_image_height=ARCH["backbone"]["rgb_image_height"])

    pred.open_scan(scan_file, image_file, l1_file, l2_file)
    pred.open_label(pred_file, False)
    u_pred_sem = remap_lut[pred.sem_label]  # remap to xentropy format
    if FLAGS.limit is not None:
      u_pred_sem = u_pred_sem[:FLAGS.limit]

    # add single scan to evaluation
    evaluator.addBatch(u_pred_sem, u_label_sem)

  # when I am done, print the evaluation
  m_accuracy = evaluator.getacc()
  m_jaccard, class_jaccard = evaluator.getIoU()

  print('Validation set:\n'
        'Acc avg {m_accuracy:.3f}\n'
        'IoU avg {m_jaccard:.3f}'.format(m_accuracy=m_accuracy,
                                         m_jaccard=m_jaccard))
  # print also classwise
  for i, jacc in enumerate(class_jaccard):
    if i not in ignore:
      print('IoU class {i:} [{class_str:}] = {jacc:.3f}'.format(
          i=i, class_str=class_strings[class_inv_remap[i]], jacc=jacc))

  # print for spreadsheet
  print("*" * 80)
  print("below can be copied straight for paper table")
  for i, jacc in enumerate(class_jaccard):
    if i not in ignore:
      sys.stdout.write('{jacc:.3f}'.format(jacc=jacc.item()))
      sys.stdout.write(",")
  sys.stdout.write('{jacc:.3f}'.format(jacc=m_jaccard.item()))
  sys.stdout.write(",")
  sys.stdout.write('{acc:.3f}'.format(acc=m_accuracy.item()))
  sys.stdout.write('\n')
  sys.stdout.flush()
