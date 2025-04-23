import hydra
from omegaconf import DictConfig
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import numpy as np
from itertools import product, permutations, chain
from functools import reduce
from typing import Iterable
from rdkit import Chem
from ast import literal_eval
import logging
from collections import defaultdict
from copy import deepcopy
from ergochemics.standardize import standardize_mol
from ergochemics.mapping import (
    operator_map_reaction,
    _m_standardize_reaction,
    rc_to_str,
    rc_to_nest
)

def clean_up_mcsa(mech_atoms: Iterable[Iterable[Iterable[int]]], rxn: str) -> tuple[list[list[int]], str]:
    '''
    De-atom map, remove ions, translate mech atoms from am nums to atom idxs
    '''
    to_rm = {
        'H',
        'Zn',
        'Mg',
        'Fe',
        'Cu',
    }

    lhs = rxn.split(">>")[0].split('.')
    rhs = rxn.split(">>")[1].split('.')
    lhs = [Chem.MolFromSmiles(x) for x in lhs]
    rhs = [Chem.MolFromSmiles(x) for x in rhs]
    lhs_mech_atoms = mech_atoms[0]
    rct_idxs = []
    de_am_smiles = []
    for mol, ams in zip(lhs, lhs_mech_atoms):
        
        # Exclude ions from lhs and rct_idxs
        if mol.GetNumAtoms() == 1 and mol.GetAtomWithIdx(0).GetSymbol() in to_rm:
            continue

        tmp = []
        for atom in mol.GetAtoms():
            if atom.GetAtomMapNum() in ams:
                tmp.append(atom.GetIdx())

            atom.SetAtomMapNum(0)

        rct_idxs.append(tmp)
        de_am_smiles.append(Chem.MolToSmiles(mol))

    # Exclude ions from rhs
    rhs = [
        Chem.MolToSmiles(x) for x in rhs
        if x.GetNumAtoms() > 1 or x.GetAtomWithIdx(0).GetSymbol() not in to_rm
    ]

    rxn = ".".join(de_am_smiles) + ">>" + ".".join(rhs)

    return rct_idxs, rxn

def rm_amns(mol: Chem.Mol) -> str:
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    return Chem.MolToSmiles(mol)

def is_strictly_balanced(lhs: Chem.Mol, rhs: Chem.Mol) -> bool:
    '''
    Check if the reaction is strictly balanced, i.e., all atom map numbers are conserved.
    '''
    lhs_elements = {f"{atom.GetSymbol()}_{atom.GetAtomMapNum()}" for atom in lhs.GetAtoms()}
    rhs_elements = {f"{atom.GetSymbol()}_{atom.GetAtomMapNum()}" for atom in rhs.GetAtoms()}
    
    return len(lhs_elements ^ rhs_elements) == 0

log = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="conf", config_name="distill_mech_labeled_reactions")
def main(cfg: DictConfig):
    min_amn = lambda mol: min(atom.GetAtomMapNum() for atom in mol.GetAtoms() if atom.GetAtomMapNum() > 0)
    sms_std = lambda smi: Chem.MolToSmiles(standardize_mol(Chem.MolFromSmiles(smi), quiet=True), ignoreAtomMapNumbers=True)

    # Load reactions
    mech_rxns = pd.read_csv(filepath_or_buffer=Path(cfg.mech_rxns), sep=",")
    mech_rxns["mech_atoms"] = mech_rxns["mech_atoms"].apply(literal_eval)

    data = []
    for _, row in mech_rxns.iterrows():
        lhs_smi_pos = defaultdict(list)
        rhs_smi_pos = defaultdict(list)
        lhs, rhs = [
            [sms_std(smi) for smi in side.split('.')]
            for side in row["smarts"].split('>>')
        ]

        for i, smi in enumerate(lhs):
            lhs_smi_pos[
                rm_amns(Chem.MolFromSmiles(smi))
            ].append(i)
            
        for i, smi in enumerate(rhs):
            rhs_smi_pos[
                rm_amns(Chem.MolFromSmiles(smi))
            ].append(i)

        # Filter out smarts that are in both sides with equal cardinalities
        unequal_cards = set()
        lhs_keep = set()
        for k, v in lhs_smi_pos.items():
            if k not in rhs_smi_pos:
                for i in v:
                    lhs_keep.add(i)
            elif len(v) != len(rhs_smi_pos[k]):
                unequal_cards.add(k)
        
        rhs_keep = set()
        for k, v in rhs_smi_pos.items():
            if k not in lhs_smi_pos:
                for i in v:
                    rhs_keep.add(i)
            elif len(v) != len(lhs_smi_pos[k]):
                unequal_cards.add(k)

        if len(lhs_keep) == 0 or len(rhs_keep) == 0:
            log.info(f"Skipping entry {row['entry_id'], row['mechanism_id']} empty lhs/rhs after removing mols unchanged in reaction")
            continue

        lhs_mols = [Chem.MolFromSmiles(lhs[i]) for i in lhs_keep]
        rhs_mols = [Chem.MolFromSmiles(rhs[i]) for i in rhs_keep]

        lhs_amns = list(
            chain(
                *[[atom.GetAtomMapNum() for atom in mol.GetAtoms()] for mol in lhs_mols]
            )
        )
        rhs_amns = list(
            chain(
                *[[atom.GetAtomMapNum() for atom in mol.GetAtoms()] for mol in rhs_mols]
            )
        )

        # Keep only those from unequal_cards that have have amn on other side of distinct mols, i.e., those in lhs/rhs mols
        for i in unequal_cards:

            for j in lhs_smi_pos[i]:
                mol = Chem.MolFromSmiles(lhs[j])
                for atom in mol.GetAtoms():
                    if atom.GetAtomMapNum() in rhs_amns:
                        lhs_mols.append(mol)
                        break

            for j in rhs_smi_pos[i]:
                mol = Chem.MolFromSmiles(rhs[j])
                for atom in mol.GetAtoms():
                    if atom.GetAtomMapNum() in lhs_amns:
                        rhs_mols.append(mol)
                        break

        # Create combined molecules
        lhs = reduce(Chem.CombineMols, lhs_mols)
        rhs = reduce(Chem.CombineMols, rhs_mols)

        if not is_strictly_balanced(lhs, rhs):
            log.info(f"Skipping entry {row['entry_id'], row['mechanism_id']} because either not balanced or amns not conserved")
            continue

        # Get reaction center amns
        lhs_amns = [atom.GetAtomMapNum() for atom in lhs.GetAtoms()]
        rhs_amns = [atom.GetAtomMapNum() for atom in rhs.GetAtoms()]
        lhs_perm = sorted([i for i in range(len(lhs_amns))], key=lambda x: lhs_amns[x])
        rhs_perm = sorted([i for i in range(len(rhs_amns))], key=lambda x: rhs_amns[x])
        lhs_A = Chem.GetAdjacencyMatrix(lhs, useBO=True)[lhs_perm][:, lhs_perm]
        rhs_A = Chem.GetAdjacencyMatrix(rhs, useBO=True)[rhs_perm][:, rhs_perm]
        rc = np.abs(lhs_A - rhs_A).sum(axis=1).nonzero()[0].tolist()
        rc_amns = [lhs_amns[lhs_perm[i]] for i in rc] # Translate back to amns; either lhs or rhs will do

        # Convert from amns in aidxs
        rc_aidxs = [[], []]
        mech_aidxs = [[], []]
        mech_amns = row['mech_atoms']
        for i, side in enumerate([lhs_mols, rhs_mols]):
            for mol in side:
                rc_aidxs[i].append([])
                mech_aidxs[i].append([])
                for atom in mol.GetAtoms():
                    amn = atom.GetAtomMapNum()
                    if amn in rc_amns:
                        rc_aidxs[i][-1].append(atom.GetIdx())
                    elif amn in mech_amns:
                        mech_aidxs[i][-1].append(atom.GetIdx())

        std_rxn = [[], []]
        std_am_rxn = [[], []]
        for i, side in enumerate([lhs_mols, rhs_mols]):
            amn_order = sorted([j for j in range(len(side))], key=lambda x: min_amn(side[x]))
            for j in amn_order:
                std_am_rxn[i].append(Chem.MolToSmiles(side[j], ignoreAtomMapNumbers=True))
                std_rxn[i].append(rm_amns(side[j]))

            rc_aidxs[i] = [rc_aidxs[i][j] for j in amn_order]
            mech_aidxs[i] = [mech_aidxs[i][j] for j in amn_order]

        std_rxn = ".".join(std_rxn[0]) + ">>" + ".".join(std_rxn[1])
        std_am_rxn = ".".join(std_am_rxn[0]) + ">>" + ".".join(std_am_rxn[1])


        data.append(
            [
                row["entry_id"], row["mechanism_id"], std_rxn, std_am_rxn,
                rc_aidxs, mech_aidxs, row["enzyme_name"], row["uniprot_id"], row["ec"]
            ]
        )

    columns = ["entry_id", "mechanism_id", "smarts", "am_smarts", "reaction_center", "mech_atoms", "enzyme_name", "uniprot_id", "ec"]
    distilled = pd.DataFrame(data, columns=columns)
    distilled["mech_atoms"] = distilled["mech_atoms"].apply(rc_to_str)
    distilled["reaction_center"] = distilled["reaction_center"].apply(rc_to_str)
    distilled.to_parquet("distilled_mech_reactions.parquet") # Save

if __name__ == "__main__":
    main()