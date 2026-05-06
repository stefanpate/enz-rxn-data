#!/bin/bash
#SBATCH -A b1039
#SBATCH -p b1039
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem=2GB
#SBATCH -t 00:05:00
#SBATCH --job-name="foo"
#SBATCH --output=/home/spn1560/enz-rxn-data/logs/out/%x_%A_%a.out
#SBATCH --error=/home/spn1560/enz-rxn-data/logs/error/%x_%A_%a.err


# Commands
ulimit -c 0
module purge
uv run python -c "import pandas; print(pandas.__version__)"
