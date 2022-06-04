# --dataset_root_directory should be where the parent folder of sequences folder
python3 ./train.py --dataset_root_directory ../../../pennovation_dataset/ -ac ./config/arch/darknet53-1024px-pennovation.yaml -dc ./config/labels/pennovation.yaml -l ../../../logs/ -p ../../../pennovation-darknet53
