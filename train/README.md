# LiDAR-Bonnetal Training

This part of the framework deals with the training of segmentation networks for point cloud data using range images.

## Tasks

- [Semantic Segmentation](tasks/semantic).
- [Panoptic Segmentation](tasks/panoptic) \[Soon\].

## Dependencies

First you need to install the nvidia driver and CUDA, so have fun!

- CUDA Installation guide: [link](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html)

- System dependencies:

  ```sh
  $ sudo apt-get update 
  $ sudo apt-get install -yqq  build-essential ninja-build \
    python3-dev python3-pip apt-utils curl git cmake unzip autoconf autogen \
    libtool mlocate zlib1g-dev python3-numpy python3-wheel wget \
    software-properties-common openjdk-8-jdk libpng-dev  \
    libxft-dev ffmpeg python3-pyqt5.qtopengl
  $ sudo updatedb
  ```

- Python dependencies

  ```sh
  $ sudo pip3 install -r requirements.txt
  ```