#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.

import argparse
import datetime
import os
import subprocess

import onnx
import torch
import yaml
import __init__ as Booger
from tasks.semantic.modules.user import User

if __name__ == '__main__':
    parser = argparse.ArgumentParser("./create_onnx.py")
    parser.add_argument(
        '--dataset', '-d',
        type=str,
        required=True,
        help='Dataset to train with. No Default',
    )
    parser.add_argument(
        '--log', '-l',
        type=str,
        default=os.path.expanduser("~") + '/logs/' +
        datetime.datetime.now().strftime("%Y-%-m-%d-%H:%M") + '/',
        help='Directory to put the predictions. Default: ~/logs/date+time'
    )
    parser.add_argument(
        '--model', '-m',
        type=str,
        required=True,
        default=None,
        help='Directory to get the trained model.'
    )
    FLAGS, unparsed = parser.parse_known_args()

    # print summary of what we will do
    print("----------")
    print("INTERFACE:")
    print("dataset", FLAGS.dataset)
    print("log", FLAGS.log)
    print("model", FLAGS.model)
    print("----------\n")
    print("Commit hash (training version): ", str(
        subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).strip()))
    print("----------\n")

    # open arch config file
    try:
        print("Opening arch config file from %s" % FLAGS.model)
        ARCH = yaml.safe_load(open(FLAGS.model + "/arch_cfg.yaml", 'r'))
    except Exception as e:
        print(e)
        print("Error opening arch yaml file.")
        quit()

    # open data config file
    try:
        print("Opening data config file from %s" % FLAGS.model)
        DATA = yaml.safe_load(open(FLAGS.model + "/data_cfg.yaml", 'r'))
    except Exception as e:
        print(e)
        print("Error opening data yaml file.")
        quit()

    # does model folder exist?
    if os.path.isdir(FLAGS.model):
        print("model folder exists! Using model from %s" % (FLAGS.model))
    else:
        print("model folder doesnt exist! Can't infer...")
        quit()

    # create user to access model
    user = User(ARCH, DATA, FLAGS.dataset, FLAGS.log, FLAGS.model)
    model = user.model

    # report model parameters
    weights_total = sum(p.numel() for p in model.parameters())
    weights_grad = sum(p.numel()
                       for p in model.parameters() if p.requires_grad)
    print("Total number of parameters: ", weights_total)
    print("Total number of parameters requires_grad: ", weights_grad)

    # convert to ONNX
    dummy_input = torch.randn(1, 5,
                              64,
                              2048, device='cuda')
    # (Pdb) proj_in.shape
    # torch.Size([1, 5, 64, 2048])
    # (Pdb) proj_range.shape (also proj_range)
    # torch.Size([1, 64, 2048])

    model = model.cuda().eval()
    onnx_path = os.path.join(FLAGS.model, "model.onnx")
    print("saving model in ", onnx_path)
    with torch.no_grad():
        torch.onnx.export(model, dummy_input, onnx_path, verbose=True)

    # check that it worked
    model_onnx = onnx.load(onnx_path)
    onnx.checker.check_model(model_onnx)
    # Print a human readable representation of the graph
    print(onnx.helper.printable_graph(model_onnx.graph))
