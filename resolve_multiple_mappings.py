from rdkit import Chem
import numpy as np
from collections import Counter
from omegaconf import DictConfig
import hydra
import pandas as pd
from pathlib import Path

def get_bond_matrix(side: list[Chem.Mol], n_atoms: int) -> tuple[np.ndarray, dict[int, str]]:
    '''
    Get the bond matrix for a list of molecules

    Args
    ----
    side: list[Chem.Mol]
        A list of molecules
    n_atoms: int
        The total number of atoms in the molecules
    
    Returns
    -------
    A: np.ndarray
        The bond matrix
    element_symbols: dict[int, str]
        A dictionary of atom indices and their corresponding element symbols
    '''
    A = np.zeros(shape=(n_atoms, n_atoms), dtype=np.float32)
    element_symbols = {}
    for mol in side:
        for bond in mol.GetBonds():
            ai, aj = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            i = mol.GetAtomWithIdx(ai).GetAtomMapNum() - 1
            j = mol.GetAtomWithIdx(aj).GetAtomMapNum() - 1
            A[i, j] = A[j, i] = bond.GetBondTypeAsDouble()

        for atom in mol.GetAtoms():
            i = atom.GetAtomMapNum() - 1
            elt_i = atom.GetSymbol()
            
            if i not in element_symbols:
                element_symbols[i] = elt_i


    return A, element_symbols

def count_bond_changes(rxn: str) -> Counter[tuple[str, float, float], int]:
    """
    Count bond changes of a certain type defined by the atomic
    constituents and the bond order on each side of the reaction

    Args
    ----
    rxn: str
        The atom-mapped reaction
    
    Returns
    -------
    
    """
    bond_changes = []
    lhs, rhs = [[Chem.MolFromSmiles(s) for s in side.split(".")] for side in rxn.split(">>")]
    nL = sum(m.GetNumAtoms() for m in lhs)
    nR = sum(m.GetNumAtoms() for m in rhs)
    
    if nL != nR:
        raise ValueError("Number of atoms in reactants and products do not match")
    
    L, element_symbols = get_bond_matrix(lhs, nL)
    R, _ = get_bond_matrix(rhs, nR)
    D = R - L
    nz = np.vstack(np.where(D != 0)).T
    for i, j in nz:
        i, j = int(i), int(j)
        atomic_constituents = "".join(sorted(element_symbols[i] + element_symbols[j]))
        bond_orders = [float(L[i, j]), float(R[i, j])]
        bond_changes.append(tuple([atomic_constituents] + bond_orders))

    bond_counts = Counter(bond_changes)
    
    return bond_counts

def does_break_cc(rxn: str) -> bool:
    """
    Check if the reaction breaks a C-C bond

    Args
    ----
    rxn: str
        The atom-mapped reaction
    
    Returns
    -------
    bool
        True if the reaction breaks a C-C bond, False otherwise
    """
    bond_cts = count_bond_changes(rxn)
    return bond_cts[("CC", 1.0, 0.0)] > 0

@hydra.main(version_base=None, config_path="conf", config_name="resolve_multiple_mappings")
def main(cfg: DictConfig):
    full = pd.read_parquet(Path(cfg.filepaths.interim_data) / "mapped_reactions.parquet")
    rule_cts = Counter(full["rule"])

    selected = []
    for name, group in full.groupby("id"):
        if len(group) == 1:
            selected.append(group.iloc[0])
        else:
            cc_breaks = group["am_smarts"].apply(does_break_cc)

            if cc_breaks.all() or not cc_breaks.any():
                selected.append(group.loc[group["rule"].map(rule_cts).idxmax()])
            else:
                not_cc = group.loc[~cc_breaks]
                selected.append(not_cc.loc[not_cc["rule"].map(rule_cts).idxmax()])
    
    selected = pd.DataFrame(selected, columns=full.columns)
    selected.reset_index(drop=True, inplace=True)

    selected.to_parquet("enzymatic_reactions.parquet")


if __name__ == "__main__":
    main()
