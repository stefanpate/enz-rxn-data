import hydra
from omegaconf import DictConfig
from ergochemics.mapping import operator_map_reaction
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import json
from tqdm import tqdm
import pandas as pd
from itertools import product

@hydra.main(version_base=None, config_path="conf", config_name="map_pathway_level_reactions")
def main(cfg: DictConfig):

    # Load reactions
    with open(Path(cfg.filepaths.raw_data) / "sprhea/sprhea_240310_v3_mapped_no_subunits.json", 'r') as f: # TODO: Change to source direct from rhea + uniprot + etc
        reactions = json.load(f)

    # Load rules
    rules = pd.read_csv(Path(cfg.filepaths.rules) / "min_rules.tsv", sep="\t")
    rules = set(rules["SMARTS"])

    chunksize = len(rules)
    ids, smarts = zip(*[(k, v['smarts']) for k, v in reactions.items()])
    tasks = list(product(smarts, rules))
    ids = [elt[0] for elt in product(ids, rules)]
    
    # Map
    with ProcessPoolExecutor() as executor:
        results = list(
            tqdm(
                executor.map(operator_map_reaction, *zip(*tasks), chunksize=chunksize),
                total=len(tasks)
            )
        )
    
    # Save
    columns = ["id", "smarts", "am_smarts", "rule", "reaction_center"] 
    data = []
    rxns, rules = zip(*tasks)
    for id, _, rule, res in zip(ids, rxns, rules, results):
        if res.did_map:
            data.append([id, res.aligned_smarts, res.atom_mapped_smarts, rule, res.reaction_center])
        
    df = pd.DataFrame(data, columns=columns)
    df.to_parquet("mapped_reactions.parquet")

if __name__ == "__main__":
    main()