import hydra
from omegaconf import DictConfig
import polars as pl
from pathlib import Path
from logging import getLogger
from time import perf_counter
from equilibrator_assets.local_compound_cache import LocalCompoundCache
from equilibrator_api.phased_reaction import PhasedReaction
from equilibrator_api import Q_, ComponentContribution
import equilibrator_assets.chemaxon as _chemaxon
from ergochemics import hash_molecule
from enz_rxn_data.pka_plugins import MolGPKA

# Monkey patch pka predictor
_predictor = MolGPKA()
_chemaxon.get_dissociation_constants = _predictor.get_dissociation_constants

def calc_dg(rxn: str) -> float:
    # TODO: wrap eQ's dg_prim_std in try/except to catch errors and return None if calculation fails
    pass

logger = getLogger(__name__)
@hydra.main(version_base=None, config_path="../conf", config_name="calc_thermo")
def main(cfg: DictConfig):
    reactions_path = Path(cfg.filepaths.processed_data) / "pathway" / "known_reactions.parquet"
    if not reactions_path.exists():
        raise FileNotFoundError(f"Known reactions file not found at {reactions_path}")
    
    krs = pl.read_parquet(reactions_path)
    kcs = pl.read_parquet(Path(cfg.filepaths.processed_data) / "pathway" / "known_compounds.parquet")
    smi2name = dict(zip(kcs['smiles'].to_list(), kcs['name'].to_list()))

    # Get unique compounds
    unique_cpds = set()
    for smarts in krs['smarts']:
        lhs, rhs = smarts.split(">>")
        for smi in lhs.split(".") + rhs.split("."):
            unique_cpds.add(smi)

    unique_cpds = pl.DataFrame(
        {
            "struct": list(unique_cpds),
            "coco_id": [hash_molecule(smi) for smi in unique_cpds],
            "name": [smi2name.get(smi, "unknown") for smi in unique_cpds],
        }
    ).to_pandas()

    logger.info(f"{len(unique_cpds)} unique compounds / {len(krs)} reactions to process")

    logger.info("Adding compounds to local cache...")
    lc = LocalCompoundCache(ccache_path=cfg.eq_cache)
    start = perf_counter()
    lc.add_compounds(
        unique_cpds,
        mol_format="smiles",
        bypass_chemaxon=True, # Still adds compound even if cxcalc fails
        save_empty_compounds=True,
    )
    end = perf_counter()
    logger.info(f"Added {len(unique_cpds)} compounds to local cache in {end - start:.2f} seconds")

    # Gather required compounds
    eq_cpds = lc.get_compounds(unique_cpds['struct'].to_list())
    smiles2cid = dict(zip(unique_cpds['struct'].to_list(), unique_cpds['coco_id'].to_list()))
    cid_2_eq_cpd = dict(zip(unique_cpds['coco_id'].to_list(), [elt.compound for elt in eq_cpds]))

    # Gather required reactions
    eq_cpd_getter = lambda cid: cid_2_eq_cpd.get(cid, None)
    rid_2_eq_rxn = {}
    for row in krs.iter_rows(named=True):
        lhs, rhs = [elt.split('.') for elt in row['smarts'].split(">>")]
        lhs = [smiles2cid[elt] for elt in lhs]
        rhs = [smiles2cid[elt] for elt in rhs]
        lhs = " + ".join(lhs)
        rhs = " + ".join(rhs)
        rxn_string = f"{lhs} = {rhs}"
        rid_2_eq_rxn[row['id']] = PhasedReaction.parse_formula(eq_cpd_getter, rxn_string)

    # Calculate path mdfs
    cc = ComponentContribution(ccache=lc.ccache)
    cc.p_h = Q_(cfg.p_h)
    cc.p_mg = Q_(cfg.p_mg)
    cc.ionic_strength = Q_(cfg.ionic_strength)
    cc.temperature = Q_(cfg.temperature)