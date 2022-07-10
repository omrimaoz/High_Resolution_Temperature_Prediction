#! /bin/sh

SBATCH --job-name=TemperaturePrediction
SBATCH --output=/vol/scratch/omri-ml/ML_Workshop2/Main.out # redirect stdout
SBATCH --error=/vol/scratch/omri-ml/ML_Workshop2/Main.err # redirect stderr
SBATCH --partition=studentbatch # (see resources section)
SBATCH --time=400 # max time (minutes)
SBATCH --signal=USR1@120 # how to end job when time’s up
SBATCH --nodes=1 # number of machines
SBATCH --ntasks=1 # number of processes
SBATCH --mem=50000 # CPU memory (MB)
SBATCH --cpus-per-task=8 # CPU cores per process
SBATCH --gpus=24 # GPUs in total


python Main.py