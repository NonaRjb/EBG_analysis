#!/usr/bin/env bash
#SBATCH --mem  20GB
#SBATCH --gres gpu:1
#SBATCH --cpus-per-task 4
#SBATCH --constrain "eowyn|galadriel|belegost|khazadum"
#SBATCH --mail-type BEGIN,END,FAIL
#SBATCH --mail-user nonar@kth.se
#SBATCH --output /Midgard/home/%u/EBG_analysis/logs/cluster_logs/%A_%a_slurm.out
#SBATCH --error  /Midgard/home/%u/EBG_analysis/logs/cluster_logs/%A_%a_slurm.err
#SBATCH --array=1-1%1

c_array=(0.01 0.1 1 10 100)
C=${c_array[$((SLURM_ARRAY_TASK_ID-1))]}

# Check job environment
echo "JOB:  ${SLURM_JOB_ID}"
echo "TASK: ${SLURM_ARRAY_TASK_ID}"
echo "HOST: $(hostname)"
echo ""
# Activate conda
source "${HOME}/miniconda3/etc/profile.d/conda.sh"
if [ "${SLURMD_NODENAME}" == "galadriel" ]; then
  conda activate cuda_11
elif [ "${SLURMD_NODENAME}" == "eowyn" ]; then
  conda activate cuda_11
else
  conda activate eegnet_pytorch
fi

python train_logistic_reg.py --subject_id 1 --data ebg4_sensor --data_type sensor_ica --modality ebg --tmin 0.06 \
--tmax 0.1 --fmin 45 --fmax 80 -c "$C" --seed 42 --save True