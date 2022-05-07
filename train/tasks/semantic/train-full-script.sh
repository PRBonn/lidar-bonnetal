# --dataset_root_directory should be where the parent folder of sequences folder
python3 ./train.py --dataset_root_directory ../../../simulated_data/ -ac ./config/arch/darknet53.yaml -dc ./config/labels/quad.yaml -l ../../../log/ -p ../../../darknet53/
