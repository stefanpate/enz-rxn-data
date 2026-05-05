#!/bin/bash
#SBATCH -A p30041
#SBATCH -p normal
#SBATCH -N 1
#SBATCH -n 30
#SBATCH --mem=0
#SBATCH -t 48:00:00
#SBATCH --job-name="map"
#SBATCH --output=/home/spn1560/enz-rxn-data/logs/out/%x_%A_%a.out
#SBATCH --error=/home/spn1560/enz-rxn-data/logs/error/%x_%A_%a.err
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --array=0-5
#SBATCH --mail-user=stefan.pate@northwestern.edu

# Args
script=/home/spn1560/enz-rxn-data/map_pathway_level_reactions.py
rxn=known_reactions_after_2015.parquet
missing_rule_cofactors=true
explicit_hs=true
rule=(
    EVODEX-Cm.csv
    EVODEX-Dm.csv
    EVODEX-Em.csv
    evodex_Cm_rules_before_2015.csv
    evodex_Dm_rules_before_2015.csv
    evodex_Em_rules_before_2015.csv

)

# Commands
ulimit -c 0
module purge
source ${UV_PROJECT_ENVIRONMENT}/bin/activate
python $script rxn_file=$rxn rule_file=${rule[$SLURM_ARRAY_TASK_ID]} missing_rule_cofactors=$missing_rule_cofactors explicit_hs=$explicit_hs
