import hydra
from omegaconf import DictConfig
from ergochemics.mapping import operator_map_reaction, rc_to_str
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import json
from tqdm import tqdm
import pandas as pd
from itertools import product

@hydra.main(version_base=None, config_path="conf", config_name="map_pathway_level_reactions")
def main(cfg: DictConfig):

    # Load reactions
    with open(Path(cfg.rxn_path), 'r') as f: # TODO: Change file type handling when change to new pull
        reactions = json.load(f)

    # Load rules
    rules = pd.read_csv(Path(cfg.rule_path), sep=",", index_col=0)
    chunksize = len(rules)

    rxn_rule_cart_prod = product(reactions.keys(), rules.index)
    tasks = [(k, reactions[k]['smarts'], rule_id, rules.loc[rule_id, "smarts"]) for k, rule_id in rxn_rule_cart_prod]
    rxn_ids, rxns, rule_ids, rules = zip(*tasks)
    
    # Map
    with ProcessPoolExecutor() as executor:
        results = list(
            tqdm(
                executor.map(operator_map_reaction, rxns, rules, chunksize=chunksize),
                total=len(tasks)
            )
        )
    
    # Save
    columns = ["rxn_id", "smarts", "am_smarts", "rule", "template_aidxs", "rule_id"] 
    data = []
    for id, rule, res, rule_id in zip(rxn_ids, rules, results, rule_ids):
        if res.did_map:
            data.append([id, res.aligned_smarts, res.atom_mapped_smarts, rule, rc_to_str(res.reaction_center), rule_id])
        
    df = pd.DataFrame(data, columns=columns)
    df.to_parquet(f"mappings_{Path(cfg.rxn_file).stem}_x_{Path(cfg.rule_file).stem}.parquet")

if __name__ == "__main__":
    main()