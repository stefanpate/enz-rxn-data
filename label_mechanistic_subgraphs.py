import hydra
from omegaconf import DictConfig
from pathlib import Path
import json
from itertools import chain
from typing import Iterable
from collections import defaultdict
from rdkit import Chem
from rdkit.Chem import rdChemReactions
from enz_rxn_data.mechanism import (
    parse_mrv,
    construct_mols,
    get_overall_reaction,
    step
)

def translate(entering_rcts: list[Chem.Mol], prev_products: tuple[Chem.Mol], lhs: Iterable[Chem.Mol]) -> tuple[dict[str, str], tuple[Chem.Mol]]:
    '''
    
    '''
    one_step_translation = {} # Maps old atom index to new atom index
    imt_rcts = [None for _ in range(len(lhs))] # Properly arranged intermediate reactants
    candidates = entering_rcts + prev_products
    remaining_candidates = {i for i in range(len(candidates))}
    for i, query in enumerate(lhs):
         for j, mol in enumerate(candidates):
            ss_match = mol.GetSubstructMatch(query)
            if len(ss_match) == max(mol.GetNumAtoms(), query.GetNumAtoms()):
                imt_rcts[i] = mol

            for match_idx, query_atom in zip(ss_match, query.GetAtoms()):
                old_aidx = mol.GetAtomWithIdx(match_idx).GetProp('mcsa_id')
                new_aidx = query_atom.GetProp('mcsa_id')
                one_step_translation[old_aidx] = new_aidx

            remaining_candidates.remove(j)
            break

    # "Lost" mass in mechanism
    for i in remaining_candidates:
        for atom in candidates[i].GetAtoms():
            one_step_translation[atom.GetProp('mcsa_id')] = None

    # Fill in the rest, e.g., residues not listed in overal reaction
    for i, patt in enumerate(lhs):
        if imt_rcts[i] is None:
            imt_rcts[i] = patt

    return one_step_translation, tuple(imt_rcts)

def transform(lhs: list[Chem.Mol], rhs: list[Chem.Mol], imt_rcts: list[Chem.Mol]) -> tuple[Chem.Mol]:
    '''
    '''
    rule = ".".join([Chem.MolToSmarts(mol) for mol in lhs]) + ">>" + ".".join([Chem.MolToSmarts(mol) for mol in rhs])
    op = rdChemReactions.ReactionFromSmarts(rule)
    products = op.RunReactants(imt_rcts)[0]

    for p in products:
        for atom in p.GetAtoms():
            props = atom.GetPropsAsDict()
            atom.SetProp('mcsa_id', f"a{props['old_mapno']}")

    return products    

@hydra.main(version_base=None, config_path="conf", config_name="label_mechanistic_subgraphs")
def main(cfg: DictConfig):
    with open(Path(cfg.filepaths.raw_mcsa) / "entries_7.json", "r") as f:
        entries = json.load(f)

    entry_id = '722'
    reaction_entry = entries[entry_id]['reaction']


    for mech in reaction_entry['mechanisms']:
        overall_lhs, overall_rhs = get_overall_reaction(reaction_entry['compounds'], Path(cfg.filepaths.raw_mcsa) / "mols")
        involved_atoms = defaultdict(dict)
        current_aidxs = defaultdict(dict)
        for i, mol in enumerate(overall_lhs):
            for atom in mol.GetAtoms():
                involved_atoms[i][atom.GetIdx()] = False
                current_aidxs[i][atom.GetIdx()] = None
        
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
        for j, rct in enumerate(overall_lhs):
            found_entry = False
            
            for i, estep in enumerate(elementary_steps):
                if found_entry:
                    break

                for k, mol in enumerate(estep[0]):
                    if found_entry:
                        break

                    ss_match = rct.GetSubstructMatch(mol)
                    if len(ss_match) == max(mol.GetNumAtoms(), rct.GetNumAtoms()):
                        entry_points[i].append(rct)
                        found_entry = True
                        break

        imt_pdts = tuple()
        for i, (estep, eflow) in enumerate(list(zip(elementary_steps, eflows))):
            entering_rcts = entry_points.get(i, [])

            # Translate
            one_step_translation, imt_rcts = translate(entering_rcts, imt_pdts, estep[0])
            for k1, v1 in current_aidxs.items():
                for k2, v2 in v1.items():
                    current_aidxs[k1][k2] = one_step_translation.get(v2, None)

            # Involved
            atoms_in_eflows = set(chain(*[chain(elt.from_, elt.to) for elt in eflow.values()])) # Collect atoms in eflows
            for k1, v1 in current_aidxs.items():
                for k2, v2 in v1.items():
                    if v2 in atoms_in_eflows:
                        involved_atoms[k1][k2] = True

            # Transform
            imt_pdts = transform(estep[0], estep[1], imt_rcts)


            
            

if __name__ == "__main__":
    main()