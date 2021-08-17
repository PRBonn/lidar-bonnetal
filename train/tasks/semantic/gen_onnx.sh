eval "$(conda shell.bash hook)"
conda activate seg
python3.7 create_onnx.py -d /home/gnardari/Documents/data/ -l ../../../quad_darknet21/ -m ../../../quad_darknet21
