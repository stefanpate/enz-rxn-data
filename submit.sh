#!/bin/bash
#SBATCH -A b1039
#SBATCH -p b1039
#SBATCH -N 1
#SBATCH -n 50
#SBATCH --mem=0
#SBATCH -t 3:00:00
#SBATCH --job-name="map_rxns"
#SBATCH --output=/home/spn1560/enz-rxn-data/logs/out/%A
#SBATCH --error=/home/spn1560/enz-rxn-data/logs/error/%A
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=stefan.pate@northwestern.edu

# Args
script=/home/spn1560/enz-rxn-data/map_pathway_level_reactions.py
rxn=sprhea_240310_v3_mapped_no_subunits.json
rule=imt_rules.csv

# Commands
ulimit -c 0
module purge
source /home/spn1560/enz-rxn-data/.venv/bin/activate
python $script rxn_file=$rxn rule_file=$rule