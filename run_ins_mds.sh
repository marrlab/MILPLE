#! /bin/bash

#SBATCH -o output_mds.txt
#SBATCH -e error_mds.txt
#SBATCH -J IBA_masks
#SBATCH -p gpu_p
#SBATCH --gres=gpu:1
#SBATCH --qos=gpu
#SBATCH -c 1
#SBATCH --mem-per-cpu=8000
#SBATCH -t 48:00:00
#SBATCH --nice=10000

source $HOME/.bashrc
conda activate matek_new

python3 -u insertion_mds.py input_iba
python3 -u insertion_mds.py feat_iba
