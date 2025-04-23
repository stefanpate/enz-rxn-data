import hydra
import logging
from omegaconf import DictConfig
from pathlib import Path
import json
from itertools import chain
from collections import defaultdict, Counter
from functools import reduce
from rdkit import Chem
import pandas as pd
from ergochemics.standardize import standardize_mol
from enz_rxn_data.mechanism import (
    parse_mrv,
    construct_mols,
    step
)

def is_strictly_balanced(lhs: list[Chem.Mol], rhs: list[Chem.Mol]) -> bool:
    '''
    Check if the reaction is strictly balanced, i.e., all atom map numbers are conserved.
    '''
    lhs = reduce(Chem.CombineMols, lhs)
    rhs = reduce(Chem.CombineMols, rhs)
    lhs_elements = {f"{atom.GetSymbol()}_{atom.GetAtomMapNum()}" for atom in lhs.GetAtoms() if atom.GetSymbol() != 'H'}
    rhs_elements = {f"{atom.GetSymbol()}_{atom.GetAtomMapNum()}" for atom in rhs.GetAtoms() if atom.GetSymbol() != 'H'}
    
    return len(lhs_elements ^ rhs_elements) == 0

def back_translate(rhs_prev: list[Chem.Mol], lhs: list[Chem.Mol], next_amn: int = 1) -> dict[int, int]:
    '''
    Gets mapping of ith atom map numbers to (i-1)th atom map numbers by comparing
    lhs_i and rhs_(i-1)
    '''
    back_translations = {}
    new = []
    influx_idxs = []
    outflux_idxs = set([i for i in range(len(rhs_prev))])
    for lidx, lmol in enumerate(lhs):
        found = False
        for idx in outflux_idxs:
            rmol = rhs_prev[idx]

            if lmol.GetNumAtoms() != rmol.GetNumAtoms(): # Cardinality different
                continue

            ss_match = lmol.GetSubstructMatch(rmol)
            if len(ss_match) == lmol.GetNumAtoms(): # Mols match

                for i, elt in enumerate(ss_match):
                    old_amn = rmol.GetAtomWithIdx(i).GetAtomMapNum()
                    new_amn = lmol.GetAtomWithIdx(elt).GetAtomMapNum()
                    back_translations[new_amn] = old_amn
                
                found = True
                outflux_idxs.remove(idx)
                break
            
            lmol_std = standardize_mol(lmol, quiet=True)
            rmol_std = standardize_mol(rmol, quiet=True)
            ss_match = lmol_std.GetSubstructMatch(rmol_std)
            
            if len(ss_match) == lmol_std.GetNumAtoms(): # Standardized mols match

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
                outflux_idxs.remove(idx)
                break
        
        if not found:
            new.append(lmol)
            influx_idxs.append(lidx)
    
    # Reindex new to disambiguate with old
    for mol in new:
        for atom in mol.GetAtoms():
            new_amn = atom.GetAtomMapNum()
            back_translations[new_amn] = next_amn
            next_amn += 1

    return back_translations, influx_idxs, outflux_idxs

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
                    overall_rhs = []
                    mech_atoms = step_mech_atoms # On 1st step, indices need not be back translated
                else:

                    try:
                        back_translations, influx_idxs, outflux_idxs = back_translate(prev_rhs, lhs, next_amn)
                    except ValueError as e:
                        log.info(f"Error back translating for entry {entry_id}, mechanism {mech['mechanism_id']}, step {estep['step_id']}: {e}")
                        misannotated_mechanism = True
                        break

                    # Translate this step lhs and rhs mols back to previous
                    for mol in lhs + rhs:
                        for atom in mol.GetAtoms():
                            new_amn = atom.GetAtomMapNum()
                            atom.SetAtomMapNum(back_translations[new_amn])

                            # Collect mech_atoms
                            if new_amn in step_mech_atoms:
                                mech_atoms.add(back_translations[new_amn])

                    overall_lhs.extend([lhs[i] for i in influx_idxs]) # Add new mols to lhs
                    overall_rhs.extend([prev_rhs[i] for i in outflux_idxs])

                next_amn += N_i
                prev_rhs = rhs

            if misannotated_mechanism:
                continue

            overall_rhs.extend(rhs) # Add last step rhs to overall rhs

            lhs_amns = Counter([atom.GetAtomMapNum() for mol in overall_lhs for atom in mol.GetAtoms()])
            rhs_amns = Counter([atom.GetAtomMapNum() for mol in overall_rhs for atom in mol.GetAtoms()])

            if any([elt > 1 for elt in lhs_amns.values()]) or any([elt > 1 for elt in rhs_amns.values()]):
                log.info(f"Entry {(entry_id, mech['mechanism_id'])} has non-unique atom map numbers in lhs or rhs")
                continue

            if not is_strictly_balanced(overall_lhs, overall_rhs):
                log.info(f"Entry {(entry_id, mech['mechanism_id'])} is not strictly balanced")
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