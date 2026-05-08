#!/bin/bash
#SBATCH -A b1039
#SBATCH -p b1039
#SBATCH -N 1
#SBATCH -n 50
#SBATCH --mem=0
#SBATCH -t 48:00:00
#SBATCH --job-name="map"
#SBATCH --output=/home/spn1560/enz-rxn-data/logs/out/%x_%A_%a.out
#SBATCH --error=/home/spn1560/enz-rxn-data/logs/error/%x_%A_%a.err
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --array=0-1
#SBATCH --mail-user=stefan.pate@northwestern.edu

# Args
script=/home/spn1560/enz-rxn-data/map_pathway_level_reactions.py
rxn=known_reactions_after_2015.parquet
missing_rule_cofactors=false
explicit_hs=true
rxn=(
    known_reactions.parquet
    known_reactions_after_2015.parquet
)
rule=(
    ehreact_rules.csv
    ehreact_rules_before_2015.csv
)

# Commands
ulimit -c 0
module purge
uv run python $script rxn_file=${rxn[$SLURM_ARRAY_TASK_ID]} rule_file=${rule[$SLURM_ARRAY_TASK_ID]} missing_rule_cofactors=$missing_rule_cofactors explicit_hs=$explicit_hs
