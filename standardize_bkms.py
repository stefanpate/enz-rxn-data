from ergochemics.standardize import standardize_rxn, hash_reaction
import polars as pl
import hydra
from omegaconf import DictConfig
from pathlib import Path
from tqdm import tqdm
import logging
import json

log = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="./conf", config_name="standardize_bkms")
def main(cfg: DictConfig):
    with open(cfg.raw_bkms) as f:
        bkms = json.load(f)

    standardized_bkms = []
    bkms_schema = pl.Schema(
        {
            "id": pl.String,
            "smarts": pl.String,
            "ec": pl.String,
            "name_rxn": pl.String,
            "name": pl.String,
            "levin_id": pl.Int32,
        }
    )

    for rxn in tqdm(bkms, desc="Standardizing BKMS reactions", total=len(bkms)):
        try:
            std_rxn = standardize_rxn(
                rxn["smiles"],
                neutralization_method="simple",
                do_remove_stereo=True,
                quiet=True,
            )
            rxn_hash = hash_reaction(std_rxn)
        except Exception as e:
            log.info(f"Error standardizing reaction '{rxn['smiles']}': {e}")
            continue
        standardized_bkms.append(
            {
                "id": rxn_hash,
                "smarts": std_rxn,
                "ec": rxn["EC_Number"],
                "name_rxn": rxn["Reaction"],
                "name": rxn["Recommended_Name"],
                "levin_id": rxn["ID"],
            }
        )

    df_bkms = pl.from_dicts(standardized_bkms, schema=bkms_schema)
    df_bkms = df_bkms.unique(subset=["id"])
    df_bkms.write_parquet("std_bkms.parquet")

if __name__ == "__main__":
    main()