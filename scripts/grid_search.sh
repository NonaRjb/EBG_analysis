#!/usr/bin/env bash
#SBATCH --mem  10GB
#SBATCH --gres gpu:0
#SBATCH --cpus-per-task 4
#SBATCH --constrain "eowyn|galadriel|belegost"
#SBATCH --mail-type BEGIN,END,FAIL
#SBATCH --mail-user nonar@kth.se
#SBATCH --output /Midgard/home/%u/EBG_analysis/logs/cluster_logs/%A_%a_slurm.out
#SBATCH --error  /Midgard/home/%u/EBG_analysis/logs/cluster_logs/%A_%a_slurm.err
#SBATCH --array=1-52%10

echo "Hello $USER! You are on node $HOSTNAME. The time is $(date)"
nvidia-smi
# Activate conda
source "${HOME}/miniconda3/etc/profile.d/conda.sh"
if [ "${SLURMD_NODENAME}" == "galadriel" ]; then
  conda activate cuda_11
elif [ "${SLURMD_NODENAME}" == "eowyn" ]; then
  conda activate cuda_11
else
  conda activate eegnet_pytorch
fi

# c_array=(0.0625 0.25 0.5 1 2 4 8 16 32 64)
# t_array=(-0.6 -0.1 0.4 0.9)
s_array=(1 2 3 4 5 6 7 8 9 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 50 51 52 53)

# C=${c_array[`expr $((SLURM_ARRAY_TASK_ID-1)) % ${#c_array[@]}`]}
# t=${t_array[`expr $((SLURM_ARRAY_TASK_ID-1)) / ${#c_array[@]}`]}
subject_id=${s_array[`expr $((SLURM_ARRAY_TASK_ID-1)) % ${#s_array[@]}`]}

# python3 -m debugpy --listen 0.0.0.0:5678 --wait-for-client train_logistic_reg.py --subject_id 0 --data ebg4 --data_type sensor_ica --modality ebg --tmin -0.5 \
# --tmax 1.5 --fmin 10 --fmax 70 -c 1 --seed 42 --save
python3 train_logistic_reg.py --subject_id "$subject_id" --data ebg4 --data_type sensor_ica --modality ebg --tmin 0.0 \
--tmax 1.0 --fmin 10 --fmax 70 -c 1.0 --model svm --task "grid_search_c" --seed 42 --save