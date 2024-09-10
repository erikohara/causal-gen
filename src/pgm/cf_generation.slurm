#!/bin/bash
####### Reserve computing resources #############
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=01:00:00
#SBATCH --mem=8GB
##SBATCH --gres=gpu:1
##SBATCH --partition=gpu-v100
##SBATCH --mail-user=erik.ohara@ucalgary.ca
##SBATCH --mail-type=ALL

####### Set environment variables ###############
module load python/anaconda3-2018.12

####### Run your script #########################
source ~/software/init-conda
conda activate causal-gen
cd /home/erik.ohara/causal-gen/src/pgm/
python -u /home/erik.ohara/causal-gen/src/pgm/cf_generation.py \
     --exp_name=simba-macaw-pgm \
     --dataset=simba \
     --data_dir=/work/forkert_lab/vibu/data/SimBA-MACAW/far_bias \
     --load_path=/home/erik.ohara/causal-gen/checkpoints/c_b/simba-pgm/checkpoint.pt \
     --vae_path=/home/erik.ohara/causal-gen/checkpoints/c_b/simba-hvae/checkpoint.pt \
     --dscm_path=/home/erik.ohara/causal-gen/checkpoints/c_b/simba-cf/10249_checkpoint.pt \
     --output_path=/work/forkert_lab/vibu/data/SimBA-MACAW/far_bias/cfs_hvae_cond_fine \
     --seed=7 \
     --epochs=300 \
     --bs=16 \
     --lr=0.001 \
     --wd=0.05 \
     --parents_x class_label bias_label \
     --eval_freq=1 
