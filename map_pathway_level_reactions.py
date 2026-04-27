import hydra
from omegaconf import DictConfig
from ergochemics.mapping import operator_map_reaction, rc_to_str
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from tqdm import tqdm
import pandas as pd
from functools import partial
from rdkit import Chem
from rdkit.Chem import AllChem
from itertools import product, combinations

def task_generator(reactions, rules):
    """Generator that yields tasks without storing them all in memory"""
    for i, rxn in reactions.iterrows():
        for j, rule in rules.iterrows():
            yield (rxn.id, rxn.smarts, rule.id, rule.smarts)


def process_task_chunk(task_chunk, missing_rule_cofactors=False):
    """Process a chunk of tasks in a single worker"""
    chunk_results = []
    for rxn_id, rxn_smarts, rule_id, rule_smarts in task_chunk:
        if missing_rule_cofactors:
            subreactions = generate_subreactions(rxn_smarts, rule_smarts)
        else:
            subreactions = [rxn_smarts]
        try:
            for rxn in subreactions:
                result = operator_map_reaction(rxn, rule_smarts)
                if result.did_map:
                    chunk_results.append([
                        rxn_id, 
                        result.aligned_smarts, 
                        result.atom_mapped_smarts, 
                        rule_smarts, 
                        rc_to_str(result.template_aidxs), 
                        rule_id
                    ])
                    break # Only one mapping per real reaction
        except Exception as e:
            print(f"Error processing task {rxn_id}, {rule_id}: {e}")
            continue
    return chunk_results


def generate_subreactions(rxn: str, operator: str) -> list[str]:
    '''Hack to cope with rule sets that don't specify balanced
    reactions because they omit cofactors. Generates all possible sub-reactions
    of cardinalityt matching the operator.'''
    _operator = AllChem.ReactionFromSmarts(operator)
    n_lhs = _operator.GetNumReactantTemplates()
    n_rhs = _operator.GetNumProductTemplates()
    rcts, pdts = [side.split(".") for side in rxn.split(">>")]
    rct_combos = list(combinations(rcts, n_lhs))
    pdt_combos = list(combinations(pdts, n_rhs))
    subreactions = list(product(rct_combos, pdt_combos))
    subreaction_smarts = []
    for rct_combo, pdt_combo in subreactions:
        rct_smarts = ".".join(rct_combo)
        pdt_smarts = ".".join(pdt_combo)
        subreaction_smarts.append(f"{rct_smarts}>>{pdt_smarts}")
    return subreaction_smarts


@hydra.main(version_base=None, config_path="conf", config_name="map_pathway_level_reactions")
def main(cfg: DictConfig):
    _process_task_chunk = partial(process_task_chunk, missing_rule_cofactors=cfg.missing_rule_cofactors)
    
    def process_batch(executor, batch, batch_num, chunk_size=50):
        """Process a single batch of tasks using chunking"""
        print(f"Processing batch {batch_num + 1} with {len(batch)} tasks...")
        
        # Split batch into chunks for workers
        chunks = [batch[i:i + chunk_size] for i in range(0, len(batch), chunk_size)]
        print(f"Split into {len(chunks)} chunks of ~{chunk_size} tasks each")
        
        # Submit chunks to workers
        futures = []
        for chunk in chunks:
            future = executor.submit(_process_task_chunk, chunk)
            futures.append(future)
        
        # Collect results as they complete
        batch_results = []
        for future in tqdm(futures, desc=f"Batch {batch_num + 1} chunks"):
            try:
                chunk_results = future.result()
                batch_results.extend(chunk_results)
            except Exception as e:
                print(f"Error processing chunk: {e}")
                continue
        
        print(f"Batch {batch_num + 1} completed with {len(batch_results)} successful mappings")
        return batch_results
    
    def process_in_batches(tasks_gen, batch_size=1000, max_workers=None, chunk_size=50):
        """Process tasks in batches to control memory usage"""
        all_results = []
        batch_count = 0
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            batch = []
            
            for task in tasks_gen:
                batch.append(task)
                
                # Process batch when it reaches the specified size
                if len(batch) >= batch_size:
                    batch_results = process_batch(executor, batch, batch_count, chunk_size)
                    all_results.extend(batch_results)
                    batch = []
                    batch_count += 1
            
            # Process any remaining tasks in the final batch
            if batch:
                batch_results = process_batch(executor, batch, batch_count, chunk_size)
                all_results.extend(batch_results)
        
        return all_results
    
    reactions = pd.read_parquet(Path(cfg.rxn_path))

    # Load rules
    rules = pd.read_csv(Path(cfg.rule_path), sep=",")
    
    print(f"Processing {len(reactions)} reactions against {len(rules)} rules")
    print(f"Total combinations: {len(reactions) * len(rules):,}")
    
    # Use generator instead of creating all tasks in memory
    tasks_gen = task_generator(reactions, rules)
    
    # Process in batches to control memory usage
    all_results = process_in_batches(
        tasks_gen, 
        batch_size=cfg.batch_size, 
        max_workers=cfg.get('max_workers', 50),
        chunk_size=cfg.get('chunk_size', 50)
    )
    
    # Create final DataFrame and save
    columns = ["rxn_id", "smarts", "am_smarts", "rule", "template_aidxs", "rule_id"]
    df = pd.DataFrame(all_results, columns=columns)
    
    output_file = f"mappings_{Path(cfg.rxn_file).stem}_x_{Path(cfg.rule_file).stem}.parquet"
    df.to_parquet(output_file, index=False)
    
    print(f"Final results saved to {output_file} with {len(df)} total mappings")

if __name__ == "__main__":
    main()