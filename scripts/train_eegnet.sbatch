#!/usr/bin/env bash
#SBATCH --mem  50GB
#SBATCH --gres gpu:1
#SBATCH --cpus-per-task 4
#SBATCH --constrain "eowyn|galadriel|belegost|khazadum"
#SBATCH --mail-type BEGIN,END,FAIL
#SBATCH --mail-user nonar@kth.se
#SBATCH --output /Midgard/home/%u/EBG_analysis/logs/cluster_logs/%A_%a_slurm.out
#SBATCH --error  /Midgard/home/%u/EBG_analysis/logs/cluster_logs/%A_%a_slurm.err
#SBATCH --array=0-0%1

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



sweep_id='ju0cw9qa'
echo 'sweep_id:' $sweep_id

project_name='EBG_Olfaction'
echo 'project_name:' $project_name

echo 'job array task:' "${SLURM_ARRAY_TASK_ID}"

entity_name='nona-phd'

export WANDB_API_KEY=d5a82a7201d64dd1120fa3be37072e9e06e382a1

wandb login ""
wandb agent $sweep_id --project $project_name --entity $entity_name