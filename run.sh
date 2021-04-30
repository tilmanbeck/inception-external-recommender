#!/bin/bash
#
#SBATCH --job-name=inception-recommender
#SBATCH --output=/ukp-storage-1/beck/slurm-inception-recommender.txt
#SBATCH --mail-user=beck@ukp.informatik.tu-darmstadt.de
#SBATCH --mail-type=ALL
#SBATCH --partition=demonstrator
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16GB
#SBATCH --gres=gpu:1

source /ukp-storage-1/beck/Repositories/inception-external-recommender/venv/bin/activate
python wsgi.py
