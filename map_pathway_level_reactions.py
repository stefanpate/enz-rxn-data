import hydra
from omegaconf import DictConfig
from ergochemics.mapping import operator_map_reaction, rc_to_str
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from tqdm import tqdm
import pandas as pd

def task_generator(reactions, rules):
    """Generator that yields tasks without storing them all in memory"""
    for i, rxn in reactions.iterrows():
        for j, rule in rules.iterrows():
            yield (rxn.id, rxn.smarts, rule.id, rule.smarts)

def process_in_batches(tasks_gen, batch_size=1000, max_workers=None):
    """Process tasks in batches to control memory usage"""
    all_results = []
    batch_count = 0
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        batch = []
        
        for task in tasks_gen:
            batch.append(task)
            
            # Process batch when it reaches the specified size
            if len(batch) >= batch_size:
                batch_results = process_batch(executor, batch, batch_count)
                all_results.extend(batch_results)
                batch = []
                batch_count += 1
        
        # Process any remaining tasks in the final batch
        if batch:
            batch_results = process_batch(executor, batch, batch_count)
            all_results.extend(batch_results)
    
    return all_results

def process_batch(executor, batch, batch_num):
    """Process a single batch of tasks"""
    print(f"Processing batch {batch_num + 1} with {len(batch)} tasks...")
    
    # Submit all tasks in the batch
    futures = []
    for rxn_id, rxn_smarts, rule_id, rule_smarts in batch:
        future = executor.submit(operator_map_reaction, rxn_smarts, rule_smarts)
        futures.append((future, rxn_id, rule_id, rule_smarts))
    
    # Collect results as they complete
    batch_results = []
    for future, rxn_id, rule_id, rule_smarts in tqdm(futures, desc=f"Batch {batch_num + 1}"):
        try:
            result = future.result()
            if result.did_map:
                batch_results.append([
                    rxn_id, 
                    result.aligned_smarts, 
                    result.atom_mapped_smarts, 
                    rule_smarts, 
                    rc_to_str(result.reaction_center), 
                    rule_id
                ])
        except Exception as e:
            print(f"Error processing task {rxn_id}, {rule_id}: {e}")
            continue
    
    print(f"Batch {batch_num + 1} completed with {len(batch_results)} successful mappings")
    return batch_results

@hydra.main(version_base=None, config_path="conf", config_name="map_pathway_level_reactions")
def main(cfg: DictConfig):

    reactions = pd.read_parquet(Path(cfg.rxn_path))[:2000]

    # Load rules
    rules = pd.read_csv(Path(cfg.rule_path), sep=",")[:100]
    
    print(f"Processing {len(reactions)} reactions against {len(rules)} rules")
    print(f"Total combinations: {len(reactions) * len(rules):,}")
    
    # Use generator instead of creating all tasks in memory
    tasks_gen = task_generator(reactions, rules)
    
    # Process in batches to control memory usage
    all_results = process_in_batches(tasks_gen, batch_size=cfg.batch_size)
    
    # Create final DataFrame and save
    columns = ["rxn_id", "smarts", "am_smarts", "rule", "template_aidxs", "rule_id"]
    df = pd.DataFrame(all_results, columns=columns)
    
    output_file = f"mappings_{Path(cfg.rxn_file).stem}_x_{Path(cfg.rule_file).stem}.parquet"
    df.to_parquet(output_file, index=False)
    
    print(f"Final results saved to {output_file} with {len(df)} total mappings")

if __name__ == "__main__":
    main()