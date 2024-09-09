#!/bin/bash
####### Reserve computing resources #############
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --mem=64GB
##SBATCH --gres=gpu:1
##SBATCH --partition=gpu-v100
#SBATCH --mail-user=erik.ohara@ucalgary.ca
#SBATCH --mail-type=ALL

####### Set environment variables ###############
module load python/anaconda3-2018.12

####### Run your script #########################
source ~/software/init-conda
conda activate causal-gen
cd /home/erik.ohara/causal-gen/src
python -u main.py \
     --exp_name=simba-hvae \
     --data_dir=/work/forkert_lab/vibu/data/SimBA-MACAW/far_bias \
     --resume=/home/erik.ohara/causal-gen/checkpoints/c_b/simba-hvae/checkpoint.pt
     --hps simba \
     --parents_x class_label bias_label \
     --context_dim=2 \
     --concat_pa \
     --cond_prior \ 
     --lr=0.001 \
     --bs=1 \
     --wd=0.05 \
     --beta=5 \
     --x_like=diag_dgauss \
     --z_max_res=96 \
     --eval_freq=4 
