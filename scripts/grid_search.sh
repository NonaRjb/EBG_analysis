#!/usr/bin/env bash
#SBATCH -A berzelius-2023-338
#SBATCH --mem  20GB
#SBATCH --gpus=0
#SBATCH --cpus-per-task 1
#SBATCH -t 05:00:00
#SBATCH --mail-type FAIL
#SBATCH --mail-user nonar@kth.se
#SBATCH --output /proj/berzelius-2023-338/users/x_nonra/EBG_analysis/logs/cluster_logs/%A_%a_slurm.out
#SBATCH --error  /proj/berzelius-2023-338/users/x_nonra/EBG_analysis/logs/cluster_logs/%A_%a_slurm.err
#SBATCH --array=1-10%10

echo "Hello $USER! You are on node $HOSTNAME. The time is $(date)"
nvidia-smi
cd /proj/berzelius-2023-338/users/x_nonra/EBG_analysis
module load Mambaforge/23.3.1-1-hpc1-bdist
mamba activate eegnet_pytorch 

c_array=(0.0625 0.25 0.5 1 2 4 8 16 32 64)
t_array=(-0.1 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 1.1 1.2 1.3 1.4 1.5)
s_array=(1 2 3 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30)

C=${c_array[`expr $((SLURM_ARRAY_TASK_ID-1)) % ${#c_array[@]}`]}
t=${t_array[`expr $((SLURM_ARRAY_TASK_ID-1)) / ${#c_array[@]}`]}
subject_id=${s_array[`expr $((SLURM_ARRAY_TASK_ID-1)) / ${#c_array[@]}`]}

# c_array=(0.0625 0.25 0.5 1 2 4 8 16 32 64)_

# python3 -m debugpy --listen 0.0.0.0:5678 --wait-for-client train_logistic_reg.py --subject_id 0 --data ebg4 --data_type sensor_ica --modality ebg --tmin -0.5 \
# --tmax 1.5 --fmin 10 --fmax 70 -c 1 --seed 42 --save
python3 train_logistic_reg.py --subject_id 0 --data ebg4 --data_type sensor_ica --modality eeg --tmin -0.5 \
--tmax 1.5 --fmin 10 --fmax 70 -c "$C" --seed 42 --save