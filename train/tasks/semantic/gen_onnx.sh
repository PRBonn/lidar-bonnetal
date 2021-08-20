eval "$(conda shell.bash hook)"
conda activate seg
CUDA_VISIBLE_DEVICES="" python3.7 create_onnx.py -d /home/gnardari/Documents/data/ -l ../../../quad_small/ -m ../../../quad_small
#CUDA_VISIBLE_DEVICES="" python3.7 create_onnx.py -d /home/gnardari/Documents/data/ -l ../../../quad_log/ -m ../../../quad_log
