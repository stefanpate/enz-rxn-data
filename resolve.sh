#!/bin/bash
#SBATCH -A b1039
#SBATCH -p b1039
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --mem=16G
#SBATCH -t 4:00:00
#SBATCH --job-name="resolve"
#SBATCH --output=/home/spn1560/enz-rxn-data/logs/out/%x_%A_%a.out
#SBATCH --error=/home/spn1560/enz-rxn-data/logs/error/%x_%A_%a.err
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --array=0-4
#SBATCH --mail-user=stefan.pate@northwestern.edu

# Args
script=/home/spn1560/enz-rxn-data/resolve_multiple_mappings.py
src_file=(
    mappings_known_reactions_x_mechinferred_dt_035_rules_before_2015_direct_mcsa_only.parquet
    mappings_known_reactions_x_mechinferred_dt_059_rules_before_2015_direct_mcsa_only.parquet
    mappings_known_reactions_x_mechinferred_dt_106_rules_before_2015_direct_mcsa_only.parquet
    mappings_known_reactions_x_mechinferred_dt_244_rules_before_2015_direct_mcsa_only.parquet
    mappings_known_reactions_x_mechinferred_dt_961_rules_before_2015_direct_mcsa_only.parquet
)

# Commands
ulimit -c 0
module purge
source /home/spn1560/enz-rxn-data/.venv/bin/activate
python $script src_file=${src_file[$SLURM_ARRAY_TASK_ID]}
