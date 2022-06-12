import subprocess
subprocess.run("python3.7 ./train.py -d /home/gnardari/Documents/Ag/segmentation/ -ac ./config/arch/darknet53.yaml -dc ./config/labels/quad.yaml -l ../../../log/ -p ../../../darknet53/")
