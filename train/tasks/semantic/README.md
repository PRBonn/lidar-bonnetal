# LiDAR-Bonnetal Semantic Segmentation Training

This part of the framework deals with the training of semantic segmentation networks for point cloud data using range images. This code allows to reproduce the experiments from the [RangeNet++](http://www.ipb.uni-bonn.de/wp-content/papercite-data/pdf/milioto2019iros.pdf) paper

_Examples of segmentation results from [SemanticKITTI](http://semantic-kitti.org) dataset:_
![ptcl](../../../pics/semantic-ptcl.gif)
![ptcl](../../../pics/semantic-proj.gif)

## Configuration files

Architecture configuration files are located at [config/arch](config/arch/)
Dataset configuration files are located at [config/labels](config/labels/)

## Apps

`ALL SCRIPTS CAN BE INVOKED WITH -h TO GET EXTRA HELP ON HOW TO RUN THEM`

### Visualization

To visualize the data (in this example sequence 00):

```sh
$ ./visualize.py -d /path/to/dataset/ -s 00
```

To visualize the predictions (in this example sequence 00):

```sh
$ ./visualize.py -d /path/to/dataset/ -p /path/to/predictions/ -s 00
```

### Training

To train a network (from scratch):

```sh
$ ./train.py -d /path/to/dataset -ac /config/arch/CHOICE.yaml -l /path/to/log
```

To train a network (from pretrained model):

```
$ ./train.py -d /path/to/dataset -ac /config/arch/CHOICE.yaml -dc /config/labels/CHOICE.yaml -l /path/to/log -p /path/to/pretrained
```

This will generate a tensorboard log, which can be visualized by running:

```sh
$ cd /path/to/log
$ tensorboard --logdir=. --port 5555
```

And acccessing [http://localhost:5555](http://localhost:5555) in your browser.

### Inference

To infer the predictions for the entire dataset:

```sh
$ ./infer.py -d /path/to/dataset/ -l /path/for/predictions -m /path/to/model
````

### Evaluation

To evaluate the overall IoU of the point clouds (of a specific split, which in semantic kitti can only be train and valid, since test is only run in our evaluation server):

```sh
$ ./evaluate_iou.py -d /path/to/dataset -p /path/to/predictions/ --split valid
```

To evaluate the border IoU of the point clouds (introduced in RangeNet++ paper):

```sh
$ ./evaluate_biou.py -d /path/to/dataset -p /path/to/predictions/ --split valid --border 1 --conn 4
```

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