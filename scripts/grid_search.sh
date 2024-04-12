#!/usr/bin/env bash
#SBATCH -A berzelius-2023-338
#SBATCH --mem  20GB
#SBATCH --cpus-per-task 1
#SBATCH -t 02:00:00
#SBATCH --mail-type FAIL
#SBATCH --mail-user nonar@kth.se
#SBATCH --output /proj/berzelius-2023-338/users/x_nonra/EBG_analysis/logs/cluster_logs/%A_%a_slurm.out
#SBATCH --error  /proj/berzelius-2023-338/users/x_nonra/EBG_analysis/logs/cluster_logs/%A_%a_slurm.err
#SBATCH --array=1-240%10

echo "Hello $USER! You are on node $HOSTNAME. The time is $(date)"
nvidia-smi
cd /proj/berzelius-2023-338/users/x_nonra/EBG_analysis
module load Mambaforge/23.3.1-1-hpc1-bdist
mamba activate eegnet_pytorch 

c_array=(0.0625 0.25 0.5 1 2 4 8 16 32 64)
s_array=(1 2 3 4 5 6 7 8 9 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25)

C=${c_array[`expr $((SLURM_ARRAY_TASK_ID-1)) % ${#c_array[@]}`]}
subject_id=${s_array[`expr $((SLURM_ARRAY_TASK_ID-1)) / ${#c_array[@]}`]}

# c_array=(0.0625 0.25 0.5 1 2 4 8 16 32 64)
# C=${c_array[$((SLURM_ARRAY_TASK_ID-1))]}

python3 train_logistic_reg.py --subject_id "${subject_id}" --data ebg4_sensor --data_type sensor_ica --modality ebg --tmin 0.06 \
--tmax 0.1 --fmin 45 --fmax 80 -c "$C" --seed 42 --save