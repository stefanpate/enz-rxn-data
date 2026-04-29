#!/bin/bash
#SBATCH -A p30041
#SBATCH -p normal
#SBATCH -N 1
#SBATCH -n 50
#SBATCH --mem=32G
#SBATCH -t 48:00:00
#SBATCH --job-name="map"
#SBATCH --output=/home/spn1560/enz-rxn-data/logs/out/%x_%A_%a.out
#SBATCH --error=/home/spn1560/enz-rxn-data/logs/error/%x_%A_%a.err
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --array=0
#SBATCH --mail-user=stefan.pate@northwestern.edu

# Args
script=/home/spn1560/enz-rxn-data/map_pathway_level_reactions.py
rxn=known_reactions.parquet
missing_rule_cofactors=true
rule=(
    retrobiocat_rules.csv

)

# Commands
ulimit -c 0
module purge
source /home/spn1560/enz-rxn-data/.venv/bin/activate
python $script rxn_file=$rxn rule_file=${rule[$SLURM_ARRAY_TASK_ID]} missing_rule_cofactors=$missing_rule_cofactors
