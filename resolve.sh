#!/bin/bash
#SBATCH -A p33041
#SBATCH -p short
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem=16G
#SBATCH -t 4:00:00
#SBATCH --job-name="resolve"
#SBATCH --output=/home/spn1560/enz-rxn-data/logs/out/%x_%A_%a.out
#SBATCH --error=/home/spn1560/enz-rxn-data/logs/error/%x_%A_%a.err
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --array=0
#SBATCH --mail-user=stefan.pate@northwestern.edu

# Args
script=/home/spn1560/enz-rxn-data/resolve_multiple_mappings.py
src_file=(
    mappings_known_reactions_x_ehreact_rules.parquet
)

# Commands
ulimit -c 0
module purge
source ${UV_PROJECT_ENVIRONMENT}/bin/activate
python $script src_file=${src_file[$SLURM_ARRAY_TASK_ID]}
