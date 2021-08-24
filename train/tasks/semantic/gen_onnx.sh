eval "$(conda shell.bash hook)"
conda activate seg

CUDA_VISIBLE_DEVICES="" python3.7 create_onnx.py -d /home/gnardari/Documents/data/ -l ../../../models/quad_bdark/ -m ../../../models/quad_bdark
# SIMBASIC DARKNET
# CUDA_VISIBLE_DEVICES="" python3.7 create_onnx.py -d /home/gnardari/Documents/data/simulated_data/ -l ../../../models/sim_quad_bdark/ -m ../../../models/sim_quad_bdark

# CUDA_VISIBLE_DEVICES="" python3.7 create_onnx.py -d /home/gnardari/Documents/data/simulated_data/ -l ../../../models/sim_quad_squeezesegv2/ -m ../../../models/sim_quad_squeezesegv2
