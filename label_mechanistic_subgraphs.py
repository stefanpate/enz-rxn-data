import hydra
import logging
from omegaconf import DictConfig
from pathlib import Path
import json
from copy import deepcopy
from itertools import chain
from collections import defaultdict
from rdkit import Chem
import pandas as pd
from ergochemics.standardize import standardize_mol
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

def neutralize_atoms(mol: Chem.Mol) -> Chem.Mol:
    mol = deepcopy(mol)
    pattern = Chem.MolFromSmarts("[+1!h0!$([*]~[-1,-2,-3,-4]),-1!$([*]~[+1,+2,+3,+4])]")
    at_matches = mol.GetSubstructMatches(pattern)
    at_matches_list = [y[0] for y in at_matches]
    for at_idx in at_matches_list:
        atom = mol.GetAtomWithIdx(at_idx)
        chg = atom.GetFormalCharge()
        hcount = atom.GetTotalNumHs()
        atom.SetFormalCharge(0)
        atom.SetNumExplicitHs(hcount - chg)
        atom.UpdatePropertyCache()
    
    return mol

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
            if lmol.GetNumAtoms() != rmol.GetNumAtoms(): # Cardinality different
                continue

            ss_match = lmol.GetSubstructMatch(rmol)
            if len(ss_match) == lmol.GetNumAtoms(): # Mols match
                # print("Found match")
                # print(Chem.MolToSmiles(rmol) ,'\n', Chem.MolToSmiles(lmol))

                for i, elt in enumerate(ss_match):
                    old_amn = rmol.GetAtomWithIdx(i).GetAtomMapNum()
                    new_amn = lmol.GetAtomWithIdx(elt).GetAtomMapNum()
                    back_translations[new_amn] = old_amn
                
                found = True
                break
            
            lmol_std = standardize_mol(lmol, quiet=True, do_neutralize=False)
            rmol_std = standardize_mol(rmol, quiet=True, do_neutralize=False)
            lmol_std = neutralize_atoms(lmol_std)
            rmol_std = neutralize_atoms(rmol_std)
            ss_match = lmol_std.GetSubstructMatch(rmol_std)
            
            if len(ss_match) == lmol_std.GetNumAtoms(): # Standardized mols match
                # print("Found match")
                # print(Chem.MolToSmiles(rmol_std) ,'\n', Chem.MolToSmiles(lmol_std))

                for i, elt in enumerate(ss_match):
                    old_amn = rmol_std.GetAtomWithIdx(i).GetAtomMapNum()
                    new_amn = lmol_std.GetAtomWithIdx(elt).GetAtomMapNum()
                    back_translations[new_amn] = old_amn

                if lmol_std.GetNumAtoms() < lmol.GetNumAtoms():
                    # TODO: Handle this case
                    raise ValueError("Unmatched atoms missing from standardized mols")
                    # new_frag = {atom.GetIdx(): atom.GetAtomMapNum() for atom in lmol.GetAtoms() if atom.GetAtomMapNum() not in back_translations.keys()}
                    # old_frag = {atom.GetIdx(): atom.GetAtomMapNum() for atom in rmol.GetAtoms() if atom.GetAtomMapNum() not in back_translations.values()}
                    # if new_frag.values() == old_frag.values():
                    #     pass

                found = True
                break
        
        if not found:
            # print("No match")
            # print(Chem.MolToSmiles(lmol))
            new.append(lmol)
    
    # Reindex new to disambiguate with old
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
    msm = lambda x : [Chem.MolFromSmiles(Chem.MolToSmiles(mol)) for mol in x]
    is_H_ion = lambda mol : all([atom.GetAtomicNum() == 1 for atom in mol.GetAtoms()]) and len(mol.GetAtoms()) == 1 # Not gas though

    # Load entries
    entries = {}
    for i in cfg.entry_batches:
        with open(Path(cfg.filepaths.raw_mcsa) / f"entries_{i}.json", "r") as f:
            entries = {**entries, **json.load(f)}

    mech_labeled_reactions = []
    columns = ['entry_id', 'mechanism_id', 'smarts', 'mech_atoms', "enzyme_name", "uniprot_id", "ec"]
    for entry_id, entry in entries.items():        
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
                
                atoms, bonds, eflows = parse_mrv(file_path)
                next_atoms, next_bonds = step(atoms, bonds, eflows)

                try:
                    lhs = construct_mols(atoms.values(), bonds.values())
                    rhs = construct_mols(next_atoms.values(), next_bonds.values())
                except Exception as e: # Catch errors in mechanism annotation
                    log.info(f"Error constructing mols for entry {entry_id}, mechanism {mech['mechanism_id']}, step {estep['step_id']}: {e}")
                    misannotated_mechanism = True
                    break

                # Capture atoms involved in mechanism before losing props to msm
                eflows = set(chain(*[chain(elt.from_, elt.to) for elt in eflows.values()]))
                step_mech_atoms = set()
                for mol in lhs:
                    for atom in mol.GetAtoms():
                        if atom.GetProp('mcsa_id') in eflows or atom.GetIntProp('coord_bond') == 1:
                            step_mech_atoms.add(atom.GetAtomMapNum())

                N_i = sum(mol.GetNumAtoms() for mol in lhs) # Count before removing explicit Hs and H ions

                # Remove explicit hydrogens
                lhs = msm(lhs)
                rhs = msm(rhs)

                # Remove H ions
                lhs = [elt for elt in lhs if not is_H_ion(elt)]
                rhs = [elt for elt in rhs if not is_H_ion(elt)]
                
                if i == 0:
                    overall_lhs = lhs
                    overall_rhs = rhs
                    mech_atoms = step_mech_atoms # On 1st step, indices need not be back translated
                else:
                    # print(f"\n{i}")

                    try:
                        back_translations = back_translate(prev_rhs, lhs, next_amn)
                    except ValueError as e:
                        log.info(f"Error back translating for entry {entry_id}, mechanism {mech['mechanism_id']}, step {estep['step_id']}: {e}")
                        misannotated_mechanism = True
                        break

                    for mol in lhs + rhs:
                        for atom in mol.GetAtoms():
                            new_amn = atom.GetAtomMapNum()
                            atom.SetAtomMapNum(back_translations[new_amn])

                            # Collect mech_atoms
                            if new_amn in step_mech_atoms:
                                mech_atoms.add(back_translations[new_amn])

                    overall_lhs = mol_union(overall_lhs, mol_left_diff(lhs, prev_rhs))
                    overall_rhs = mol_union(mol_left_diff(overall_rhs, lhs), rhs)

                next_amn += N_i
                prev_rhs = rhs

            if misannotated_mechanism:
                continue

            # TODO: Figure out strict limit on atom map numbers thing. See notepad
            try:
                # ignoreAtomMapNumbers option required on lhs to canonicalize SMILES in a way that doesn't depend on atom map numbers
                smarts = ".".join([Chem.MolToSmiles(mol, ignoreAtomMapNumbers=True) for mol in overall_lhs]) + ">>" +\
                    ".".join([Chem.MolToSmiles(mol, ignoreAtomMapNumbers=True) for mol in overall_rhs])
                mech_labeled_reactions.append([entry_id, mech['mechanism_id'], smarts, list(mech_atoms), entry.get("enzyme_name"), entry.get("reference_uniprot_id"), entry["reaction"].get("ec")])
            except:
                continue
    # Save
    df = pd.DataFrame(mech_labeled_reactions, columns=columns)
    df.to_csv("mech_labeled_reactions.csv", index=False)

if __name__ == "__main__":
    main()