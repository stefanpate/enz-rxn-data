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

def process_task_chunk(task_chunk):
    """Process a chunk of tasks in a single worker"""
    chunk_results = []
    for rxn_id, rxn_smarts, rule_id, rule_smarts in task_chunk:
        try:
            result = operator_map_reaction(rxn_smarts, rule_smarts)
            if result.did_map:
                chunk_results.append([
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
    return chunk_results

def process_batch(executor, batch, batch_num, chunk_size=50):
    """Process a single batch of tasks using chunking"""
    print(f"Processing batch {batch_num + 1} with {len(batch)} tasks...")
    
    # Split batch into chunks for workers
    chunks = [batch[i:i + chunk_size] for i in range(0, len(batch), chunk_size)]
    print(f"Split into {len(chunks)} chunks of ~{chunk_size} tasks each")
    
    # Submit chunks to workers
    futures = []
    for chunk in chunks:
        future = executor.submit(process_task_chunk, chunk)
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