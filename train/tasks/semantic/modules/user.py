#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import imp
import yaml
import time
from PIL import Image
import __init__ as booger
import collections
import copy
import cv2
import os
import numpy as np
import pdb

from tasks.semantic.modules.segmentator import *
from tasks.semantic.postproc.KNN import KNN
import torch.quantization
import time


class User():
    def __init__(self, ARCH, DATA, datadir, logdir, modeldir, should_quantize=False):
        # parameters
        self.ARCH = ARCH
        self.DATA = DATA
        self.datadir = datadir
        self.logdir = logdir
        self.modeldir = modeldir
        self.use_rgb_image = None
        self.rgb_image_width = None
        self.rgb_image_height = None
        self.calibration_sequences = []

        if "use_rgb_image" in self.ARCH["backbone"]:
            self.use_rgb_image = self.ARCH["backbone"]["use_rgb_image"]
        if "rgb_image_width" in self.ARCH["backbone"]:
            self.rgb_image_width = self.ARCH["backbone"]["rgb_image_width"]
        if "rgb_image_height" in self.ARCH["backbone"]:
            self.rgb_image_height = self.ARCH["backbone"]["rgb_image_height"]
        if "calibration" in self.DATA["split"]:
            self.calibration_sequences = self.DATA["split"]["calibration"]
        # get the data
        parserModule = imp.load_source("parserModule",
                                       booger.TRAIN_PATH + '/tasks/semantic/dataset/' +
                                       self.DATA["name"] + '/parser.py')
        self.parser = parserModule.Parser(root=self.datadir,
                                          train_sequences=self.DATA["split"]["train"],
                                          valid_sequences=self.DATA["split"]["valid"],
                                          test_sequences=self.DATA["split"]["test"],
                                          labels=self.DATA["labels"],
                                          color_map=self.DATA["color_map"],
                                          learning_map=self.DATA["learning_map"],
                                          learning_map_inv=self.DATA["learning_map_inv"],
                                          sensor=self.ARCH["dataset"]["sensor"],
                                          max_points=self.ARCH["dataset"]["max_points"],
                                          batch_size=1,
                                          workers=self.ARCH["train"]["workers"],
                                          gt=True,
                                          shuffle_train=False,
                                          calibration_sequences=self.calibration_sequences,
                                          use_rgb_image=self.use_rgb_image,
                                          rgb_image_width=self.rgb_image_width,
                                          rgb_image_height=self.rgb_image_height)

        # concatenate the encoder and the head
        with torch.no_grad():
            self.model = Segmentator(self.ARCH,
                                     self.parser.get_n_classes(),
                                     self.modeldir, should_quantize=should_quantize, strict=True)

        # use knn post processing?
        self.post = None
        if self.ARCH["post"]["KNN"]["use"]:
            self.post = KNN(self.ARCH["post"]["KNN"]["params"],
                            self.parser.get_n_classes())

        # GPU?
        self.gpu = False
        self.model_single = self.model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("Infering in device: ", self.device)
        if torch.cuda.is_available() and torch.cuda.device_count() > 0 and not should_quantize:
            cudnn.benchmark = True
            cudnn.fastest = True
            self.gpu = True
            self.model.cuda()

    def infer(self):
        # do train set
        self.infer_subset(loader=self.parser.get_train_set(),
                          to_orig_fn=self.parser.to_original)

        # do valid set
        self.infer_subset(loader=self.parser.get_valid_set(),
                          to_orig_fn=self.parser.to_original)
        # do test set
        self.infer_subset(loader=self.parser.get_test_set(),
                          to_orig_fn=self.parser.to_original)

        print('Finished Infering')

        return

    def infer_test(self, neval_batches=-1):
        # do train set
        # self.infer_subset(loader=self.parser.get_train_set(),
        #                  to_orig_fn=self.parser.to_original)

        # do valid set
        # self.infer_subset(loader=self.parser.get_valid_set(),
        #                  to_orig_fn=self.parser.to_original)
        # do test set
        self.infer_subset(loader=self.parser.get_test_set(),
                          to_orig_fn=self.parser.to_original, neval_batches=neval_batches)

        print('Finished Infering')

        return

    def infer_calibrate(self):
        self.model.eval()
        print('SIZE OF MODEL BEFORE QUANTIZATION')
        self.model.print_size_of_model()
        self.model.fuse_model()
        # self.model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
        self.model.qconfig = torch.quantization.default_qconfig
        self.model.CRF.qconfig = None
        torch.quantization.prepare(self.model, inplace=True)

        self.infer_subset(loader=self.parser.get_train_set(),
                          to_orig_fn=self.parser.to_original, neval_batches=10)

        self.model = torch.quantization.convert(self.model, inplace=False)
        print("model quantization done!")
        print('SIZE OF MODEL AFTER QUANTIZATION')
        self.model.print_size_of_model()

    def infer_subset(self, loader, to_orig_fn, neval_batches=-1):
        # switch to evaluate mode
        # self.model.eval()

        # empty the cache to infer in high res
        if self.gpu:
            torch.cuda.empty_cache()
        cnt = 0
        with torch.no_grad():
            end = time.time()
            forward_times = []
            for i, (
            proj_in, proj_mask, _, _, path_seq, path_name, p_x, p_y, proj_range, unproj_range, _, _, _, _, npoints,
            proj_rgb, proj_fusion_labels, image) in enumerate(loader):
                # first cut to rela size (batch size one allows it)
                p_x = p_x[0, :npoints]
                p_y = p_y[0, :npoints]
                proj_range = proj_range[0, :npoints]
                unproj_range = unproj_range[0, :npoints]
                path_seq = path_seq[0]
                path_name = path_name[0]

                if self.gpu:
                    proj_in = proj_in.cuda()
                    proj_mask = proj_mask.cuda()
                    p_x = p_x.cuda()
                    p_y = p_y.cuda()
                    if self.use_rgb_image:
                        image = image.cuda()
                    if self.post:
                        proj_range = proj_range.cuda()
                        unproj_range = unproj_range.cuda()

                # compute output
                if self.use_rgb_image:
                    ts_start = time.time()
                    proj_output = self.model([proj_in, image], proj_mask)
                else:
                    ts_start = time.time()
                    proj_output = self.model(proj_in, proj_mask)
                ts_end = time.time()
                ts_forward = ts_end - ts_start
                forward_times.append(ts_forward)
                print("forward pass time for seq", path_seq, "scan", path_name,
                      "in", ts_forward, "sec")
                proj_argmax = proj_output[0].argmax(dim=0)
                # pdb.set_trace()
                if self.post:
                    # knn postproc
                    unproj_argmax = self.post(proj_range,
                                              unproj_range,
                                              proj_argmax,
                                              p_x,
                                              p_y)
                else:
                    # put in original pointcloud using indexes
                    unproj_argmax = proj_argmax[p_y, p_x]

                # measure elapsed time
                if torch.cuda.is_available():
                    torch.cuda.synchronize()

                print("Infered seq", path_seq, "scan", path_name,
                      "in", time.time() - end, "sec")
                end = time.time()
                cnt += 1

                # save scan
                # get the first scan in batch and project scan
                pred_np = unproj_argmax.cpu().numpy()
                pred_np = pred_np.reshape((-1)).astype(np.int32)

                # map to original label
                pred_np = to_orig_fn(pred_np)

                # save scan
                path = os.path.join(self.logdir, "sequences",
                                    path_seq, "predictions", path_name)
                pred_np.tofile(path)
                # np.save(path, pred_np)
                if cnt >= neval_batches and neval_batches != -1:
                    break
            forward_times = np.array(forward_times, dtype=np.float64)
            print("mean forward time in sec", np.nanmean(forward_times))
            print("std forward time in sec", np.nanstd(forward_times))
            print("mean forward time in milli sec", np.nanmean(forward_times * 1000))
            print("std forward time in milli sec", np.nanstd(forward_times * 1000))
