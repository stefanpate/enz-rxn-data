import hydra
from omegaconf import DictConfig
import pandas as pd
from collections import defaultdict
from pathlib import Path

@hydra.main(version_base=None, config_path="conf", config_name="uniquify_ni_rules")
def main(cfg: DictConfig):

    ni_rules = pd.read_csv(cfg.src_file, sep="\t")
    ni_sma_id = defaultdict(list)
    for _, row in ni_rules.iterrows():
        ni_sma_id[row["SMARTS"]].append(row["Name"])
    
    unique_rules = set(ni_rules["SMARTS"])
    unique_rules = pd.DataFrame(
        data=[(i, smarts, ni_sma_id[smarts]) for i, smarts in enumerate(unique_rules)],
        columns=["id", "smarts", "ni_ids"]
    )

    if cfg.src_file == "ni_min_rules.tsv":
        unique_rules.to_csv(
            "rc_plus_0_rules.csv",
            sep=",",
            index=False
        )
    else:
        unique_rules.to_csv(
            f"{"_".join(Path(cfg.src_file).stem.split('_')[1:])}.csv",
            sep=",",
            index=False
        )

if __name__ == "__main__":
    main()