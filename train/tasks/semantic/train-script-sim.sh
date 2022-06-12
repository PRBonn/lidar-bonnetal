eval "$(conda shell.bash hook)"
conda activate seg
python3.7 ./train.py -d /home/gnardari/Documents/data/simulated_data -ac ./config/arch/quad/quad_sim_squeezesegv2.yaml -dc ./config/labels/quad_binary.yaml -l ../../../models/sim_quad_squeezesegv2/
# python3.7 ./train.py -d /home/gnardari/Documents/data/simulated_data -ac ./config/arch/quad_sim_squeezesegv2.yaml -dc ./config/labels/quad_binary.yaml -l ../../../models/sim_quad_squeeze/ -p ../../../squeezesegV2
#python3.7 ./train.py -d /home/gnardari/Documents/data/ -ac ./config/arch/quad_small.yaml -dc ./config/labels/quad_binary.yaml -l ../../../quad_small/
#python3.7 ./train.py -d /home/gnardari/Documents/data/ -ac ./config/arch/quad_darknet21.yaml -dc ./config/labels/quad_binary.yaml -l ../../../quad_log/ -p ../../../darknet21/
