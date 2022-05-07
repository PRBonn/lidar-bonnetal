
# Specific instructions for our use:

## Some helpful additional utils:

1. **Convert the original labeled tree-only .pcd files into .npy label files**: 

see convert_pcds.py

2. **Split training and test data**: 

see train_test_split.py

_Example usage:_

Step 1: Make sure all your data are in the range_images folder:

For this, you can get one example from our provided dataset (see below) and then run the following steps:

Download data from https://drive.google.com/drive/folders/16MSQf-tdD1QTYpVVtElMy1enAQv7hFAR?usp=sharing

Put data into simulated_data folder (folder structure looks like this: lidar-bonnetal->simulated_data->sequences)

Copy all pcd files to range_images directory using the following commands:
```
cd ~/lidar-bonnetal
cp -r simulated_data/sequences/*/*.pcd range_images/
```

Copy all label files to range_images directory using the following commands:
```
mkdir range_images/labels
cp -r simulated_data/sequences/*/labels/*.npy range_images/labels
```

Step 2: Training and validation splitting set by running the following commands:
```
cd ~/lidar-bonnetal
python train_test_split.py
```


## Step-by-step instructions with our example data

### Step: preparing data
Create folders named range_images and simulated_data in this root directory 

Download data from https://drive.google.com/drive/folders/16MSQf-tdD1QTYpVVtElMy1enAQv7hFAR?usp=sharing

Put data into simulated_data folder (folder structure looks like this: lidar-bonnetal->simulated_data->sequences)

### Step: Installing dependencies
Option 1. Using pip:
```
cd train
pip install -r requirements.txt
cd ..
pip freeze > pip_requirements.txt
pip install -r pip_requirements.txt 
```
Note: this will take a while (> 1 hour for me)

Option 2. Using conda: see conda_requirements.txt

Troubleshooting:

1. if you run into pypcd issues, this might be helpful: https://github.com/dimatura/pypcd/issues/28

2. TO BE CONTINUED...

### Step: start training:
python train/tasks/semantic/train.py






# Official instructions for LiDAR-Bonnetal

Semantic Segmentation of point clouds using range images.

Developed by [Andres Milioto](http://www.ipb.uni-bonn.de/people/andres-milioto/), [Jens Behley](http://www.ipb.uni-bonn.de/people/jens-behley/), [Ignacio Vizzo](http://www.ipb.uni-bonn.de/people/ignacio-vizzo/), and [Cyrill Stachniss](http://www.ipb.uni-bonn.de/people/cyrill-stachniss/)

_Examples of segmentation results from [SemanticKITTI](http://semantic-kitti.org) dataset:_
![ptcl](pics/semantic-ptcl.gif)
![ptcl](pics/semantic-proj.gif)

## Description

This code provides code to train and deploy Semantic Segmentation of LiDAR scans, using range images as intermediate representation. The training pipeline can be found in [/train](train/). We will open-source the deployment pipeline soon.

## Pre-trained Models

### [SemanticKITTI](http://semantic-kitti.org)

- [squeezeseg](http://www.ipb.uni-bonn.de/html/projects/bonnetal/lidar/semantic/models/squeezeseg.tar.gz)
- [squeezeseg + crf](http://www.ipb.uni-bonn.de/html/projects/bonnetal/lidar/semantic/models/squeezeseg-crf.tar.gz)
- [squeezesegV2](http://www.ipb.uni-bonn.de/html/projects/bonnetal/lidar/semantic/models/squeezesegV2.tar.gz)
- [squeezesegV2 + crf](http://www.ipb.uni-bonn.de/html/projects/bonnetal/lidar/semantic/models/squeezesegV2-crf.tar.gz)
- [darknet21](http://www.ipb.uni-bonn.de/html/projects/bonnetal/lidar/semantic/models/darknet21.tar.gz)
- [darknet53](http://www.ipb.uni-bonn.de/html/projects/bonnetal/lidar/semantic/models/darknet53.tar.gz)
- [darknet53-1024](http://www.ipb.uni-bonn.de/html/projects/bonnetal/lidar/semantic/models/darknet53-1024.tar.gz)
- [darknet53-512](http://www.ipb.uni-bonn.de/html/projects/bonnetal/lidar/semantic/models/darknet53-512.tar.gz)

To enable kNN post-processing, just change the boolean value to `True` in the `arch_cfg.yaml` file parameter, inside the model directory.
  
## Predictions from Models

### [SemanticKITTI](http://semantic-kitti.org)

These are the predictions for the train, validation, and test sets. The performance can be evaluated for the training and validation set, but for test set evaluation a submission to the benchmark needs to be made (labels are not public).

No post-processing:
- [squeezeseg](http://www.ipb.uni-bonn.de/html/projects/bonnetal/lidar/semantic/predictions/squeezeseg.tar.gz)
- [squeezeseg + crf](http://www.ipb.uni-bonn.de/html/projects/bonnetal/lidar/semantic/predictions/squeezeseg-crf.tar.gz)
- [squeezesegV2](http://www.ipb.uni-bonn.de/html/projects/bonnetal/lidar/semantic/predictions/squeezesegV2.tar.gz)
- [squeezesegV2 + crf](http://www.ipb.uni-bonn.de/html/projects/bonnetal/lidar/semantic/predictions/squeezesegV2-crf.tar.gz)
- [darknet21](http://www.ipb.uni-bonn.de/html/projects/bonnetal/lidar/semantic/predictions/darknet21.tar.gz)
- [darknet53](http://www.ipb.uni-bonn.de/html/projects/bonnetal/lidar/semantic/predictions/darknet53.tar.gz)
- [darknet53-1024](http://www.ipb.uni-bonn.de/html/projects/bonnetal/lidar/semantic/predictions/darknet53-1024.tar.gz)
- [darknet53-512](http://www.ipb.uni-bonn.de/html/projects/bonnetal/lidar/semantic/predictions/darknet53-512.tar.gz)

With k-NN processing:
- [squeezeseg](http://www.ipb.uni-bonn.de/html/projects/bonnetal/lidar/semantic/predictions/squeezeseg-knn.tar.gz)
- [squeezesegV2](http://www.ipb.uni-bonn.de/html/projects/bonnetal/lidar/semantic/predictions/squeezesegV2-knn.tar.gz)
- [darknet53](http://www.ipb.uni-bonn.de/html/projects/bonnetal/lidar/semantic/predictions/darknet53-knn.tar.gz)
- [darknet21](http://www.ipb.uni-bonn.de/html/projects/bonnetal/lidar/semantic/predictions/darknet21-knn.tar.gz)
- [darknet53-1024](http://www.ipb.uni-bonn.de/html/projects/bonnetal/lidar/semantic/predictions/darknet53-1024-knn.tar.gz)
- [darknet53-512](http://www.ipb.uni-bonn.de/html/projects/bonnetal/lidar/semantic/predictions/darknet53-512-knn.tar.gz)

## License

### LiDAR-Bonnetal: MIT

Copyright 2019, Andres Milioto, Jens Behley, Cyrill Stachniss. University of Bonn.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

### Pretrained models: Model and Dataset Dependent

The pretrained models with a specific dataset maintain the copyright of such dataset.

## Citations

If you use our framework, model, or predictions for any academic work, please cite the original [paper](http://www.ipb.uni-bonn.de/wp-content/papercite-data/pdf/milioto2019iros.pdf), and the [dataset](http://semantic-kitti.org).

```
@inproceedings{milioto2019iros,
  author    = {A. Milioto and I. Vizzo and J. Behley and C. Stachniss},
  title     = {{RangeNet++: Fast and Accurate LiDAR Semantic Segmentation}},
  booktitle = {IEEE/RSJ Intl.~Conf.~on Intelligent Robots and Systems (IROS)},
  year      = 2019,
  codeurl   = {https://github.com/PRBonn/lidar-bonnetal},
  videourl  = {https://youtu.be/wuokg7MFZyU},
}
```

```
@inproceedings{behley2019iccv,
  author    = {J. Behley and M. Garbade and A. Milioto and J. Quenzel and S. Behnke and C. Stachniss and J. Gall},
  title     = {{SemanticKITTI: A Dataset for Semantic Scene Understanding of LiDAR Sequences}},
  booktitle = {Proc. of the IEEE/CVF International Conf.~on Computer Vision (ICCV)},
  year      = {2019}
}
```
