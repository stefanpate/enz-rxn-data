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

def standardize_de_atom_map(mol: Chem.Mol) -> str:
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    
    mol = standardize_mol(mol, quiet=True)

    return Chem.MolToSmiles(mol)

log = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="conf", config_name="distill_mech_labeled_reactions")
def main(cfg: DictConfig):
    min_amn = lambda mol: min(atom.GetAtomMapNum() for atom in mol.GetAtoms() if atom.GetAtomMapNum() > 0)

    # Load reactions
    mech_rxns = pd.read_csv(filepath_or_buffer=Path(cfg.mech_rxns), sep=",")
    mech_rxns["mech_atoms"] = mech_rxns["mech_atoms"].apply(literal_eval)

    # Remove atom map numbers & translate mech_atoms from am nums to atom idxs
    mech_atom_idxs = []
    de_am_rxns = []
    for _, row in mech_rxns.iterrows():
        lhs_smi_pos = defaultdict(list)
        rhs_smi_pos = defaultdict(list)
        lhs, rhs = [side.split('.') for side in row["smarts"].split('>>')]

        for i, smi in enumerate(lhs):
            lhs_smi_pos[
                standardize_de_atom_map(Chem.MolFromSmiles(smi))
            ].append(i)
            
        for i, smi in enumerate(rhs):
            rhs_smi_pos[
                standardize_de_atom_map(Chem.MolFromSmiles(smi))
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

        lhs_mols = [
            Chem.MolFromSmiles(lhs[i] for i in lhs_keep),
        ]
        rhs_mols = [
            Chem.MolFromSmiles(rhs[i] for i in rhs_keep),
        ]

        lhs_amns = list(chain(*[
            [atom.GetAtomMapNum() for atom in mol.GetAtoms()]
            for mol in lhs_mols
        ]))
        rhs_amns = list(chain(*[
            [atom.GetAtomMapNum() for atom in mol.GetAtoms()]
            for mol in rhs_mols
        ]))

        del lhs_amns, rhs_amns, lhs_keep, rhs_keep

        # Keep only those from unequal_cards that have have amn on other side of distinct mols, i.e., those in lhs/rhs mols
        for i in unequal_cards:

            for j in lhs_smi_pos[i]:
                found = False
                mol = Chem.MolFromSmiles(lhs[j])
                for atom in mol.GetAtoms():
                    if atom.GetAtomMapNum() in rhs_amns:
                        lhs_mols.append(mol)
                        found = True
                        break

                if found:
                    break

            for j in rhs_smi_pos[i]:
                found = False
                mol = Chem.MolFromSmiles(rhs[j])
                for atom in mol.GetAtoms():
                    if atom.GetAtomMapNum() in lhs_amns:
                        rhs_mols.append(mol)
                        found = True
                        break

                if found:
                    break

        # Get reaction center amns
        lhs = reduce(Chem.CombineMols, lhs_mols)
        rhs = reduce(Chem.CombineMols, rhs_mols)
        lhs_A = Chem.GetAdjacencyMatrix(lhs, useBO=True)
        rhs_A = Chem.GetAdjacencyMatrix(rhs, useBO=True)
        rc_amns = np.abs(lhs_A - rhs_A).sum(axis=1).nonzero()[0].tolist()

        # Convert from amns in aidxs
        rc_aidxs = [[], []]
        mech_aidxs = [[], []]
        mech_amns = row['mech_atoms']
        for i, side in enumerate([lhs_mols, rhs_mols]):
            for mol in side:
                for atom in mol.GetAtoms():
                    amn = atom.GetAtomMapNum()
                    if amn in rc_amns:
                        rc_aidxs[i].append(atom.GetIdx())
                    elif amn in mech_amns[i]:
                        mech_aidxs[i].append(atom.GetIdx())

        std_rxn = [[], []]
        for i, side in enumerate([lhs_mols, rhs_mols]):
            amn_order = sorted([j for j in range(len(side))], key=lambda x: min_amn(side[x]))
            for j in amn_order:
                std_rxn[i].append(
                    Chem.MolToSmiles(
                        standardize_de_atom_map(side[j]),
                        ignoreAtomMapNumbers=True
                        )
                )

            rc_aidxs[i] = [rc_aidxs[i][j] for j in amn_order]
            mech_aidxs[i] = [mech_aidxs[i][j] for j in amn_order]

        std_rxn = ".".join(std_rxn[0]) + ">>" + ".".join(std_rxn[1])



        

        


        
    columns = ["rxn_id", "smarts", "am_smarts", "rule", "reaction_center", "rule_id"] 
    data = []
    for rxn_id, _, rule, res, rule_id in zip(rxn_ids, rxns, rules, results, rule_ids):
        if res.did_map:
            data.append([rxn_id, res.aligned_smarts, res.atom_mapped_smarts, rule, res.reaction_center, rule_id])

    compiled.reset_index(drop=True, inplace=True)
    compiled = compiled[
        [
            "entry_id", "mechanism_id", "smarts", "am_smarts", "rule", "reaction_center",
            "mech_atoms", "enzyme_name", "uniprot_id", "ec", "rule_id"
        ]
    ]

    compiled["mech_atoms"] = compiled["mech_atoms"].apply(lambda x: rc_to_str([x, [[]]]))
    compiled["reaction_center"] = compiled["reaction_center"].apply(lambda x: rc_to_str(x))

    compiled.to_parquet("mapped_mech_labeled_reactions.parquet") # Save

if __name__ == "__main__":
    main()