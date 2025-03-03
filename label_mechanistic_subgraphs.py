import hydra
from omegaconf import DictConfig
from pathlib import Path
import json
from itertools import chain
from typing import Iterable
from collections import defaultdict
from rdkit import Chem
from rdkit.Chem import rdChemReactions
from rdkit.Chem.rdMolDescriptors import CalcNumHeavyAtoms
import pandas as pd
from ergochem.standardize import standardize_mol
from enz_rxn_data.mechanism import (
    parse_mrv,
    construct_mols,
    get_overall_reaction,
    step
)

def get_standard_match(mol: Chem.Mol, query: Chem.Mol) -> tuple[bool, tuple[int, ...]]:
    try:
        _mol = standardize_mol(mol, quiet=True)
        _query = standardize_mol(query, quiet=True)
    except Exception as e:
        print(f"Could not standardize: {e}")
        ss_match = mol.GetSubstructMatch(query)
        did_match = len(ss_match) == max(mol.GetNumAtoms(), query.GetNumAtoms())
    else:
        ss_match = _mol.GetSubstructMatch(_query)
        did_match = len(ss_match) == max(_mol.GetNumAtoms(), _query.GetNumAtoms())

    return did_match, ss_match

def translate(entering_rcts: list[Chem.Mol], prev_products: tuple[Chem.Mol], lhs: Iterable[Chem.Mol]) -> tuple[dict[str, str], tuple[Chem.Mol]]:
    '''
    
    '''
    one_step_translation = {} # Maps old atom index to new atom index
    imt_rcts = [None for _ in range(len(lhs))] # Properly arranged intermediate reactants
    candidates = entering_rcts + prev_products
    remaining_candidates = {i for i in range(len(candidates))}
    for i, query in enumerate(lhs):
         for j, mol in enumerate(candidates):
            
            if j not in remaining_candidates:
                continue
            
            did_match, ss_match = get_standard_match(mol, query)
            if did_match: # Exact match and not already used
                imt_rcts[i] = query # Safer from variation in protonation states & explicit / implicit hydrogens

                for match_idx, query_atom in zip(ss_match, query.GetAtoms()):
                    old_aidx = mol.GetAtomWithIdx(match_idx).GetProp('mcsa_id')
                    new_aidx = query_atom.GetProp('mcsa_id')
                    one_step_translation[old_aidx] = new_aidx

                remaining_candidates.remove(j)
                break

    # Molecules not required this step
    for i in remaining_candidates:
        for atom in candidates[i].GetAtoms():
            mcsa_id = atom.GetProp('mcsa_id')

            if mcsa_id.startswith('pre_'): # Has not yet entered mechanism
                continue
            else: # No longer needed to describe mechanism
                one_step_translation[mcsa_id] = None

    # Fill in the rest, e.g., residues not listed in overal reaction
    for i, patt in enumerate(lhs):
        if imt_rcts[i] is None:
            imt_rcts[i] = patt

    return one_step_translation, tuple(imt_rcts)

def transform(lhs: list[Chem.Mol], rhs: list[Chem.Mol], imt_rcts: list[Chem.Mol]) -> list[Chem.Mol]:
    '''
    '''
    rule = ".".join([Chem.MolToSmarts(mol) for mol in lhs]) + ">>" + ".".join([Chem.MolToSmarts(mol) for mol in rhs])
    op = rdChemReactions.ReactionFromSmarts(rule)
    products = op.RunReactants(imt_rcts)[0]

    for p in products:
        for atom in p.GetAtoms():
            props = atom.GetPropsAsDict()
            atom.SetProp('mcsa_id', f"a{props['old_mapno']}")

    return list(products)    

@hydra.main(version_base=None, config_path="conf", config_name="label_mechanistic_subgraphs")
def main(cfg: DictConfig):
    entries = {}
    for i in cfg.entry_batches:
        with open(Path(cfg.filepaths.raw_mcsa) / f"entries_{i}.json", "r") as f:
            entries = {**entries, **json.load(f)}

    mech_labeled_reactions = []
    columns = ['entry_id', 'mechanism_id', 'smarts', 'mech_atoms']
    for entry_id in ['49', '722', '219']:
        reaction_entry = entries[entry_id]['reaction']

        for mech in reaction_entry['mechanisms']:
            try:
                overall_lhs, overall_rhs = get_overall_reaction(reaction_entry['compounds'], Path(cfg.filepaths.raw_mcsa) / "mols")
                involved_atoms = defaultdict(dict)
                current_aidxs = defaultdict(dict)
                for i, mol in enumerate(overall_lhs):
                    for atom in mol.GetAtoms():
                        involved_atoms[i][atom.GetIdx()] = False
                        atom.SetProp('mcsa_id', f"pre_{i}_{atom.GetIdx()}") # Init atom prop pre mcsa steps w/ unique atom id
                        current_aidxs[i][atom.GetIdx()] = atom.GetProp('mcsa_id')
                
                elementary_steps = []
                eflows = []
                for estep in mech['steps']:
                    file_path = Path(cfg.filepaths.raw_mcsa) / "mech_steps" / f"{entry_id}_{mech['mechanism_id']}_{estep['step_id']}.mrv"
                    atoms, bonds, meflows = parse_mrv(file_path)
                    lhs = construct_mols(atoms.values(), bonds.values())
                    next_atoms, next_bonds = step(atoms, bonds, meflows)
                    rhs = construct_mols(next_atoms.values(), next_bonds.values())
                    elementary_steps.append((lhs, rhs))
                    eflows.append(meflows)

                # Reactants may enter at different elementary steps
                entry_points = defaultdict(list) # Maps estep -> list[entering rcts]
                remaining_overall_lhs = {i for i in range(len(overall_lhs))}
                for i, estep in reversed(list(enumerate(elementary_steps))):
                    if len(remaining_overall_lhs) == 0:
                        break

                    for k, mol in enumerate(estep[0]):
                        for j, rct in enumerate(overall_lhs):
                            if j in remaining_overall_lhs:
                                did_match, _ = get_standard_match(rct, mol)
                                if did_match:
                                    entry_points[i].append(rct)
                                    remaining_overall_lhs.remove(j)
                                    break

                imt_pdts = []
                for i, (estep, eflow) in enumerate(list(zip(elementary_steps, eflows))):
                    entering_rcts = entry_points.get(i, [])

                    # Translate
                    one_step_translation, imt_rcts = translate(entering_rcts, imt_pdts, estep[0])
                    for k1, v1 in current_aidxs.items():
                        for k2, v2 in v1.items():
                            if v2 in one_step_translation:
                                current_aidxs[k1][k2] = one_step_translation[v2]

                    # Involved
                    atoms_in_eflows = set(chain(*[chain(elt.from_, elt.to) for elt in eflow.values()])) # Collect atoms in eflows
                    for k1, v1 in current_aidxs.items():
                        for k2, v2 in v1.items():
                            if v2 in atoms_in_eflows:
                                involved_atoms[k1][k2] = True

                    # Transform
                    imt_pdts = transform(estep[0], estep[1], imt_rcts)

                smarts = ".".join([Chem.MolToSmiles(mol) for mol in overall_lhs]) + ">>" + ".".join([Chem.MolToSmiles(mol) for mol in overall_rhs])
                mech_atoms = []
                for i in sorted(involved_atoms.keys()):
                    mech_atoms.append([j for j in sorted(involved_atoms[i].keys()) if involved_atoms[i][j]])
                mech_labeled_reactions.append([entry_id, mech['mechanism_id'], smarts, mech_atoms])
            except Exception as e:
                print(f"Error processing entry {entry_id}, mechanism {mech['mechanism_id']}: {e}")
                continue

    df = pd.DataFrame(mech_labeled_reactions, columns=columns)
    df.to_csv("mech_labeled_reactions.csv", index=False)

if __name__ == "__main__":
    main()