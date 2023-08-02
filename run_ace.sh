#!/bin/sh
#SBATCH --account=visualai    # Specify VisualAI
#SBATCH --nodes=1             # nodes requested
#SBATCH --ntasks=1            # tasks requested
#SBATCH --cpus-per-task=8    # Specify the number of CPUs your task will need.
#SBATCH --gres=gpu:rtx_3090:1          # the number of GPUs requested
#SBATCH --mem=50G             # memory 
#SBATCH -o temp/%j_%x_run.txt         # send stdout to outfile
#SBATCH -e temp/%j_%x_err.txt         # send stderr to errfile
#SBATCH -t 72:00:00           # time requested in hour:minute:second

python ace_run.py \
--n_samples 800
