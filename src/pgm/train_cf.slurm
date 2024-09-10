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
python -u train_cf.py \
     --exp_name=simba-cf \
     --dataset=simba \
     --data_dir=/work/forkert_lab/vibu/data/SimBA-MACAW/far_bias \
     --pgm_path=/home/erik.ohara/causal-gen/checkpoints/c_b/simba-pgm/checkpoint.pt \
     --predictor_path=/home/erik.ohara/causal-gen/checkpoints/c_b/simba-sup_aux/checkpoint.pt \
     --vae_path=/home/erik.ohara/causal-gen/checkpoints/c_b/simba-hvae/checkpoint.pt \
     --seed=7 \
     --bs=8 \
     --lr=0.001 \
     --wd=0.05 \
     --parents_x class_label bias_label \
     --eval_freq=1 
