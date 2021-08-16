import subprocess
subprocess.run("python ./train.py -d ../../../Training_Tom/ -ac ./config/arch/darknet53.yaml -dc ./config/labels/quad.yaml -l ../../../log/ -p ../../../darknet53/")
