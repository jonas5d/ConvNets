
import argparse
import os
import datetime
import random
import subprocess
import numpy as np

parser = argparse.ArgumentParser()

parser.add_argument("folds",type=str,
                    help='The folds that should be run')
parser.add_argument("logFolder",type=str,
                    help='The Folder that shall be used for output')
parser.add_argument("epochs",type=str,
                    help='Number of epochs.')
parser.add_argument("input_workflow",type=str,
                    help='textfile containing the workflow for the processing of input data')
parser.add_argument("net_id",type=str,
                    help='The Net ID')
parser.add_argument("--seed", nargs=1, type=int,
                    help='Seed Number random if not supplied')
parser.add_argument('--cudaLayers', action='store_true')
parser.add_argument('--sbatch', action='store_true')

args = parser.parse_args()

seed = args.seed[0] if args.seed else random.randint(0, 999999)   

#create the log directory
now = datetime.datetime.now()
stamp = now.strftime("%y%m_%d_%H%M_%S")
directory = args.logFolder + "/" + stamp
if not os.path.exists(directory):
  os.makedirs(directory)
else:
  raise ValueError("In the directory a folder with this name already exists!")

params = "THEANO_FLAGS='lib.cnmem=1' python run_net.py "
if args.sbatch:
  params += "$SLURM_ARRAY_TASK_ID "
else:
  params += args.folds + " "

params += directory + " " + args.epochs + " " + str(seed) + " " + args.input_workflow + " " + args.net_id

if args.cudaLayers:
 params += " --cudaLayers"

if args.sbatch:
  scriptString = "#! /bin/bash\n"+params+"\n#SBATCH --time=05-23:00:00\n#SBATCH --gres=gpu:1"#\n#SBATCH --output "+directory
  with open("cluster_runner.sh", "wb") as text_file:
    text_file.write(scriptString)
  subprocess.Popen("chmod u+x cluster_runner.sh", shell=True)
  folds = np.array(args.folds.split(',')).astype(np.int32)
  scmd = "sbatch --array " + str(min(folds)) + "-" + str(max(folds)) + " ./cluster_runner.sh"
  subprocess.Popen(scmd, shell=True)
else:
  print params
  logPath = directory + "/logs.txt"
  with open(logPath,"wb") as out:
    subprocess.Popen(params, shell=True,stdout=out)
