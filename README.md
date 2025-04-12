# Enzymatic reaction data

This repository contains code to download and format enzymatic reaction data from a variety of sources including M-CSA, Rhea, UniProt.

## Install

```
curl -LsSf https://astral.sh/uv/install.sh | sh # install uv
uv python install 3.13 # Install uv
git clone git@github.com:stefanpate/enz-rxn-data.git
cd enz-rxn-data
uv sync
# Set conf/filepaths/filepaths.yaml field "repo" to repo location
```

## Usage

### M-CSA

1. Run download_mcsa.py.
2. Run fill_missing_mcsa_mols.py to fill in with ChEBI mols.
3. (Optional) run label_mechanistic_subraphs.py to reaction subgraphs involved in mechanism.