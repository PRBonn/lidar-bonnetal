eval "$(conda shell.bash hook)"
conda activate seg
python3.7 ./train.py -d /home/gnardari/Documents/data/ -ac ./config/arch/quad_squeezesegv2.yaml -dc ./config/labels/quad_binary.yaml -l ../../../models/quad_squeeze/ -p ../../../models/sim_quad_squeeze
