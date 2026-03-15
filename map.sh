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
#SBATCH --array=0-12
#SBATCH --mail-user=stefan.pate@northwestern.edu

# Args
script=/home/spn1560/enz-rxn-data/map_pathway_level_reactions.py
rxn=known_reactions.parquet
rule=(
    imt_rules_before_2015.csv
    mechinferred_dt_005_rules_before_2015.csv
    mechinferred_dt_009_rules_before_2015.csv
    mechinferred_dt_021_rules_before_2015.csv
    mechinferred_dt_069_rules_before_2015.csv
    mechinferred_dt_932_rules_before_2015.csv
    mechinformed_rules_before_2015.csv
    rc_plus_0_rules_before_2015.csv
    rc_plus_1_rules_before_2015.csv
    rc_plus_2_rules_before_2015.csv
    rc_plus_3_rules_before_2015.csv
    rc_plus_4_rules_before_2015.csv
    rdchiral_rules_before_2015.csv
)

# Commands
ulimit -c 0
module purge
source /home/spn1560/enz-rxn-data/.venv/bin/activate
python $script rxn_file=$rxn rule_file=${rule[$SLURM_ARRAY_TASK_ID]}
