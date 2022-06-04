# --dataset_root_directory should be where the parent folder of sequences folder
python3 ./train.py --dataset_root_directory ../../../pennovation_dataset_jackle/ -ac ./config/arch/darknet53-1024px-pennovation-jackle.yaml -dc ./config/labels/pennovation-jackle.yaml -l ../../../log/ -p ../../../pennovation-darknet53-jackle
