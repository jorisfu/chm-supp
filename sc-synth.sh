#!/bin/bash -eux
#SBATCH --job-name=synthetic_data_clustering
#SBATCH --account=sci-renard
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=joris.funk@student.hpi.de
#SBATCH --partition=cpu-batch # -p
#SBATCH --cpus-per-task=128 # -c
#SBATCH --mem=32gb
#SBATCH --time=10:05:00
#SBATCH --output=job_test_%j.log # %j is job id

eval "$(conda shell.bash hook)"
conda activate chm-experiments
python3 /sc/home/joris.funk/chm-supp/experiments/missingness/synthetic.py
