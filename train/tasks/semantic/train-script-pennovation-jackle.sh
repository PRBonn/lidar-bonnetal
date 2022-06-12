# --dataset_root_directory should be where the parent folder of sequences folder
python3 ./train.py --dataset_root_directory ../../../pennovation_dataset_jackle/ -ac ./config/arch/darknet-smallest-1024px-pennovation-jackle.yaml -dc ./config/labels/pennovation-jackle.yaml -l ../../../logs-jackle/ -p ../../../pennovation-darknet-smallest-jackle
