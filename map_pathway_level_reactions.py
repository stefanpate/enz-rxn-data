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
    with open(Path(cfg.input_path), 'r') as f: # TODO: Change file type handling when change to new pull
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
            data.append([id, res.aligned_smarts, res.atom_mapped_smarts, rule, rc_to_str(res.reaction_center)])
        
    df = pd.DataFrame(data, columns=columns)
    df.to_parquet(f"mapped_{Path(cfg.src_file).stem}.parquet")

if __name__ == "__main__":
    main()