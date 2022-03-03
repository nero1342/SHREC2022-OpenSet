#!/bin/bash
#SBATCH --job-name=openset            # create a short name for your job
#SBATCH --output=/home/nero/SHREC2022/openset/slurm.out      # create a output file
#SBATCH --error=/home/nero/SHREC2022/openset/slurm.err       # create a error file
#SBATCH --partition=batch          # choose partition
#SBATCH --gres=gpu:1              # gpu count
#SBATCH --ntasks=1                 # total number of tasks across all nodes
#SBATCH --nodes=1                  # node count
#SBATCH --cpus-per-task=4          # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --time=3-0:00:00

echo   Date              = $(date)
echo   Hostname          = $(hostname -s)
echo   Working Directory = $(pwd)
echo   Number of Nodes Allocated      = $SLURM_JOB_NUM_NODES
echo   Number of Tasks Allocated      = $SLURM_NTASKS
echo   Number of Cores/Task Allocated = $SLURM_CPUS_PER_TASK

# Reconfigure HOME_PATH
cd /home/nero/MediaEval2021/Medico/Medico/
# pwd

python sleep.py