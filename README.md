# Enzymatic reaction data

This repository contains code to download and format enzymatic reaction data from a variety of sources including M-CSA, Rhea, UniProt.

## Install

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh # install uv
uv python install 3.13 # Install uv
git clone git@github.com:stefanpate/enz-rxn-data.git
cd enz-rxn-data
uv sync
# Set conf/filepaths/filepaths.yaml field "repo" to repo location
```

## Usage

### M-CSA

1. download_mcsa.py: downloads entries from Mechanistic and Catalytic Site Atlas
2. label_mechanistic_subgraphs.py: converts CML-encoded mechanisms into SMARTS and labels atoms involved in the mechanism
3. distill_mech_labeled_reactions.py: extracts overall reactants and products (not residues or cofactors), balances reaction and reverses it

Usage: 

```bash
python download_mcsa.py
python label_mechanistic_subgraphs.py
python distill_mech_labled_reactions.py
```

### Pathway-level enzymatic reactions

1. Download enzymes from uniprot. Note the below will be resource intensive. Recommend chunking into multiple pages or using UniProt ui and selecting all the fields below

```bash
cd data/raw/pathway
curl -o uniprot_reviewed.tsv.gz "https://rest.uniprot.org/uniprotkb/stream?compressed=true&fields=accession%2Creviewed%2Cprotein_name%2Corganism_name%2Clength%2Csequence%2Cec%2Ccc_catalytic_activity%2Cprotein_existence%2Cdate_created&format=tsv&query=%28reviewed%3Atrue%29"
gunzip uniprot_reviewed.tsv.gz
```

2. Download reaction files from [Rhea](https://www.rhea-db.org/help/download) to data/raw/pathway
- `rhea-reaction-smiles.tsv`
- `rhea-directions.tsv`
- `rhea2ec.tsv`
- `rhea-chebi-smiles.tsv`
- `chebiId_name.tsv`

3. Run combine_rhea_uniprot.py: Outputs parquet files for known reactions, enzymes and compounds. Reaction and molecule SMILES are standardized

```bash
python combine_rhea_uniprot.py
```

### Mapping rules to reactions
1. Put rules in artifacts/rules in a csv with required columns id | smarts

