import hydra
from omegaconf import DictConfig
import pandas as pd
from collections import defaultdict

@hydra.main(version_base=None, config_path="conf", config_name="uniquify_min_rules")
def main(_: DictConfig):

    ni_rules = pd.read_csv("ni_rules.tsv", sep="\t")
    ni_sma_id = defaultdict(list)
    for _, row in ni_rules.iterrows():
        ni_sma_id[row["SMARTS"]].append(row["Name"])
    
    min_rules = set(ni_rules["SMARTS"])
    min_rules = pd.DataFrame(
        data=[(i, smarts, ni_sma_id[smarts]) for i, smarts in enumerate(min_rules)],
        columns=["id", "smarts", "ni_ids"]
    )
    min_rules.to_csv(
        "min_rules.csv",
        sep=",",
        index=False
    )

if __name__ == "__main__":
    main()