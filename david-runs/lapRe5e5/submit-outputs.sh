#!/bin/bash
#SBATCH --job-name=lap-worm-tracking
#SBATCH --output=job_%j.log
#SBATCH --error=job_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=06:00:00

python3 -u worm_tracking.py "v0*.vtk"
