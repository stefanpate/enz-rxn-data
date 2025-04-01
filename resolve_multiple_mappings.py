from collections import Counter
from omegaconf import DictConfig
import hydra
import pandas as pd
from pathlib import Path
from enz_rxn_data.mapping import does_break_cc

@hydra.main(version_base=None, config_path="conf", config_name="resolve_multiple_mappings")
def main(cfg: DictConfig):
    full = pd.read_parquet(Path(cfg.input_path))
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

    selected.to_parquet(f"{cfg.src_file.split('_')[1]}")

if __name__ == "__main__":
    main()
