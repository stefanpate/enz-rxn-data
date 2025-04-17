import hydra
import logging
from omegaconf import DictConfig
from pathlib import Path
import json
from copy import deepcopy
from itertools import chain
from typing import Iterable
from collections import defaultdict
from rdkit import Chem
from rdkit.Chem import rdChemReactions
import pandas as pd
from ergochemics.standardize import standardize_mol
from ergochemics.mapping import rc_to_str
from enz_rxn_data.mechanism import (
    parse_mrv,
    construct_mols,
    step
)

def is_balanced(lhs: list[Chem.Mol], rhs: list[Chem.Mol]) -> bool:
    if len(lhs) == 0 or len(rhs) == 0:
        return False
    
    lhs_atoms = defaultdict(int)
    rhs_atoms = defaultdict(int)

    for mol in lhs:
        for atom in mol.GetAtoms():
            lhs_atoms[atom.GetSymbol()] += 1

    for mol in rhs:
        for atom in mol.GetAtoms():
            rhs_atoms[atom.GetSymbol()] += 1

    return lhs_atoms == rhs_atoms

def standardize_de_atom_map(mol: Chem.Mol) -> str:
    mol = deepcopy(mol)
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    
    mol = standardize_mol(mol, quiet=True)

    return Chem.MolToSmiles(mol)

def back_translate(rhs_prev: list[Chem.Mol], lhs: list[Chem.Mol], next_amn: int = 1) -> dict[int, int]:
    '''
    Gets mapping of ith atom map numbers to (i-1)th atom map numbers by comparing
    lhs_i and rhs_(i-1)
    '''
    back_translations = {}
    new = []
    for lmol in lhs:
        found = False
        for rmol in rhs_prev:
            if lmol.GetNumAtoms() != rmol.GetNumAtoms():
                continue

            ss_match = lmol.GetSubstructMatch(rmol)
            if len(ss_match) == lmol.GetNumAtoms():
                print("Found match")
                print(Chem.MolToSmiles(lmol) ,'\n', Chem.MolToSmiles(rmol))
                for i, elt in enumerate(ss_match):
                    old_amn = rmol.GetAtomWithIdx(i).GetAtomMapNum()
                    new_amn = lmol.GetAtomWithIdx(elt).GetAtomMapNum()
                    back_translations[new_amn] = old_amn
                
                found = True
                break
        
        if not found:
            print("No match")
            print(Chem.MolToSmiles(lmol))
            new.append(lmol)
    
    for mol in new:
        for atom in mol.GetAtoms():
            new_amn = atom.GetAtomMapNum()
            back_translations[new_amn] = next_amn
            next_amn += 1

    return back_translations

def mol_union(left: list[Chem.Mol], right: list[Chem.Mol]) -> list[Chem.Mol]:
    '''
    Takes union of two lists of mols
    '''
    left = {Chem.MolToSmiles(mol) for mol in left}
    right = {Chem.MolToSmiles(mol) for mol in right}

    lur = left | right
    return [Chem.MolFromSmiles(k) for k in lur]

def mol_left_diff(left: list[Chem.Mol], right: list[Chem.Mol]) -> list[Chem.Mol]:
    '''
    Takes left difference of two lists of mols
    '''
    left = {Chem.MolToSmiles(mol) for mol in left}
    right = {Chem.MolToSmiles(mol) for mol in right}

    ldr = left - right
    return [Chem.MolFromSmiles(k) for k in ldr]

log = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="conf", config_name="label_mechanistic_subgraphs")
def main(cfg: DictConfig):
    msm = lambda x : [Chem.MolFromSmiles(Chem.MolToSmiles(mol)) for mol in x] # TODO: Do I need this?
    is_hydrogen = lambda mol : all([atom.GetAtomicNum() == 1 for atom in mol.GetAtoms()]) and len(mol.GetAtoms()) == 1
    # msm = lambda x: x

    # Load entries
    entries = {}
    for i in cfg.entry_batches:
        with open(Path(cfg.filepaths.raw_mcsa) / f"entries_{i}.json", "r") as f:
            entries = {**entries, **json.load(f)}

    mech_labeled_reactions = []
    columns = ['entry_id', 'mechanism_id', 'smarts', 'mech_atoms', "enzyme_name", "uniprot_id", "ec"]
    for entry_id, entry in {'711': entries['711']}.items():#entries.items():        
        for mech in entry['reaction']['mechanisms']:
            
            if not mech['is_detailed']:
                continue

            # Assemble steps and electron flows
            misannotated_mechanism = False
            next_amn = 1
            for i, estep in enumerate(mech['steps']):
                file_path = Path(cfg.filepaths.raw_mcsa) / "mech_steps" / f"{entry_id}_{mech['mechanism_id']}_{estep['step_id']}.mrv"
                
                if not file_path.exists():
                    log.info(f"File {file_path} does not exist")
                    misannotated_mechanism = True
                    break
                
                atoms, bonds, meflows = parse_mrv(file_path)
                next_atoms, next_bonds = step(atoms, bonds, meflows)

                try:
                    lhs = construct_mols(atoms.values(), bonds.values())
                    rhs = msm(construct_mols(next_atoms.values(), next_bonds.values()))
                    
                    meflows = set(chain(*[chain(elt.from_, elt.to) for elt in meflows.values()]))
                    step_mech_atoms = set()
                    for mol in lhs.GetAtoms():
                        for atom in mol.GetAtoms():
                            if atom.GetProp('mcsa_id') in meflows or atom.GetIntProp('coord_bond') == 1:
                                step_mech_atoms.add(atom.GetAtomMapNum())
                
                except Exception as e: # Catch errors in mechanism annotation
                    log.info(f"Error constructing mols for entry {entry_id}, mechanism {mech['mechanism_id']}, step {estep['step_id']}: {e}")
                    misannotated_mechanism = True
                    break

                if i == 0:
                    overall_lhs = lhs
                    overall_rhs = rhs
                    mech_atoms = step_mech_atoms
                else:
                    print(f"\n{i}")
                    back_translations = back_translate(prev_rhs, lhs, next_amn)

                    for mol in lhs + rhs:
                        for atom in mol.GetAtoms():
                            new_amn = atom.GetAtomMapNum()
                            atom.SetAtomMapNum(back_translations[new_amn])

                            if new_amn in step_mech_atoms:
                                mech_atoms.add(back_translations[new_amn])

                    # print(sorted([Chem.MolToSmiles(mol) for mol in overall_lhs]), '\n')
                    overall_lhs = mol_union(overall_lhs, mol_left_diff(lhs, prev_rhs))
                    # print(sorted([Chem.MolToSmiles(mol) for mol in overall_lhs]), '\n')
                    overall_rhs = mol_union(mol_left_diff(overall_rhs, lhs), rhs)

                next_amn += sum(mol.GetNumAtoms() for mol in lhs)
                prev_rhs = rhs
                prev_lhs = lhs

            if misannotated_mechanism:
                continue

            # ignoreAtomMapNumbers option required on lhs to canonicalize SMILES in a way that doesn't depend on atom map numbers
            smarts = ".".join([Chem.MolToSmiles(mol, ignoreAtomMapNumbers=True) for mol in overall_lhs]) + ">>" + ".".join([Chem.MolToSmiles(mol) for mol in overall_rhs])
            mech_labeled_reactions.append([entry_id, mech['mechanism_id'], smarts, rc_to_str([mech_atoms, [[]]]), entry.get("enzyme_name"), entry.get("reference_uniprot_id"), entry["reaction"].get("ec")])

    # Save
    df = pd.DataFrame(mech_labeled_reactions, columns=columns)
    df.to_csv("mech_labeled_reactions.csv", index=False)

if __name__ == "__main__":
    main()