# Enzymatic reaction data

This repository contains code to download and format enzymatic reaction data from a variety of sources including M-CSA, Rhea, UniProt.

## Install

Download repo via git.

Create virtual environment and install dependencies.

```
uv sync
```

## Usage

### M-CSA

1. Run download_mcsa.py.
2. Run fill_missing_mcsa_mols.py to fill in with ChEBI mols.
3. (Optional) run label_mechanistic_subraphs.py to reaction subgraphs involved in mechanism.