#!/bin/bash
####### Reserve computing resources #############
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=24:00:00
#SBATCH --mem=16GB
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu-v100
#SBATCH --mail-user=erik.ohara@ucalgary.ca
#SBATCH --mail-type=ALL

####### Set environment variables ###############
module load python/anaconda3-2018.12

####### Run your script #########################
source ~/software/init-conda
conda activate causal-gen
cd /home/erik.ohara/causal-gen/src/pgm
python -u /home/erik.ohara/causal-gen/src/pgm/train_pgm.py \
     --exp_name=simba-sup_aux \
     --dataset=simba \
     --data_dir=/work/forkert_lab/vibu/data/SimBA-MACAW/far_bias \
     --seed=7 \
     --epochs=1000 \
     --setup=sup_aux \
     --bs=64 \
     --lr=0.001 \
     --wd=0.05 \
     --parents_x class_label bias_label \
     --eval_freq=1 
