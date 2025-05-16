from collections import Counter
from omegaconf import DictConfig
import hydra
from hydra.utils import call
import pandas as pd
from pathlib import Path
from ergochemics.mapping import rc_to_nest, rc_to_str

@hydra.main(version_base=None, config_path="conf", config_name="resolve_multiple_mappings")
def main(cfg: DictConfig):
    full = pd.read_parquet(Path(cfg.input_path))
    rule_cts = Counter(full["rule"])
    full["template_aidxs"] = full["template_aidxs"].apply(rc_to_nest)

    selected = []
    for _, group in full.groupby("rxn_id"):
        selected.append(
            call(cfg.resolver, group, rule_cts)
        )
    
    selected = pd.DataFrame(selected, columns=full.columns)
    selected.reset_index(drop=True, inplace=True)
    selected["template_aidxs"] = selected["template_aidxs"].apply(rc_to_str)

    selected.to_parquet(f"mapped_{"_".join(cfg.src_file.split('_')[1:])}")

if __name__ == "__main__":
    main()
