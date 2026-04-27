#!/bin/bash
#SBATCH -A b1039
#SBATCH -p b1039
#SBATCH -N 1
#SBATCH -n 50
#SBATCH --mem=32G
#SBATCH -t 48:00:00
#SBATCH --job-name="map"
#SBATCH --output=/home/spn1560/enz-rxn-data/logs/out/%x_%A_%a.out
#SBATCH --error=/home/spn1560/enz-rxn-data/logs/error/%x_%A_%a.err
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --array=0-4
#SBATCH --mail-user=stefan.pate@northwestern.edu

# Args
script=/home/spn1560/enz-rxn-data/map_pathway_level_reactions.py
rxn=known_reactions.parquet
rule=(
    mechinferred_dt_035_rules_before_2015_direct_mcsa_only.csv
    mechinferred_dt_059_rules_before_2015_direct_mcsa_only.csv
    mechinferred_dt_106_rules_before_2015_direct_mcsa_only.csv
    mechinferred_dt_244_rules_before_2015_direct_mcsa_only.csv
    mechinferred_dt_961_rules_before_2015_direct_mcsa_only.csv
)

# Commands
ulimit -c 0
module purge
source /home/spn1560/enz-rxn-data/.venv/bin/activate
python $script rxn_file=$rxn rule_file=${rule[$SLURM_ARRAY_TASK_ID]}
