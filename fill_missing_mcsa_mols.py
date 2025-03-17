import hydra
from omegaconf import DictConfig
from rdkit import Chem
from pathlib import Path
import json
import logging

log = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="conf", config_name="fill_mcsa_mols")
def main(cfg: DictConfig) -> None:
    """
    Fill in missing MCSA molecules from ChEBI.
    """

    # Read chebi mols into dict
    chebi_mols = {}
    for mol in Chem.SDMolSupplier(Path(cfg.chebi_mols_path)):
        if mol is None:
            continue
        chebi_mols[mol.GetProp("ChEBI ID").removeprefix("CHEBI:")] = mol

    # Iterate over mcsa entries look for missing mols
    # and check chebi. Save to mols dir if found
    for i in cfg.entry_batches:
        with open(Path(cfg.filepaths.raw_mcsa) / f"entries_{i}.json", "r") as f:
            mcsa_entries = json.load(f)
        
        for id, entry in mcsa_entries.items():
            for cpd in entry["reaction"]["compounds"]:
                if cpd["mol_file"] == "":
                    chebi_mol = chebi_mols.get(cpd["chebi_id"])

                    if chebi_mol is not None:
                        # Save mol to file
                        mol_path = Path(cfg.filepaths.mcsa_mols) / f"{cpd['chebi_id']}.mol"
                        with open(mol_path, 'w') as f:
                            f.write(Chem.MolToMolBlock(chebi_mol))
                        
                    else:
                        log.info(f"Invalid ChEBI mol for ID {cpd['chebi_id']}")
                else:
                    log.info(f"Mol file already exists for ID {cpd['chebi_id']}")
    
    log.info("Finished filling missing MCSA molecules.")


if __name__ == "__main__":
    main()