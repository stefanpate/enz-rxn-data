import requests
import hydra
from omegaconf import DictConfig
from pathlib import Path
from typing import Iterable
from functools import wraps
import json

def request_guard(func: callable) -> callable:
    """
    Decorator to handle HTTP request errors.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except requests.exceptions.RequestException as e:
            print(f"An error occurred: {e}")
            return None
    return wrapper

@request_guard
def fetch_json(url: str) -> dict | None:
    response = requests.get(url)
    response.raise_for_status()  # Raise an exception for HTTP errors
    return response.json()

@request_guard
def fetch_mrv(url: str) -> dict:
    response = requests.get(url)
    response.raise_for_status()  # Raise an exception for HTTP errors
    return response.text

@request_guard
def fetch_mol(url: str) -> dict:
    response = requests.get(url)
    response.raise_for_status()  # Raise an exception for HTTP errors
    return response.text

def extract_entries(results: list[dict], key: str, fields: Iterable[str]) -> dict:
    '''
    Returns a dictionary of the fields requested from the response
    '''
    stuff = {}
    for result in results:
        if key in result:
            stuff[result[key]] = {field: result.get(field) for field in fields}
    return stuff

def extract_mechanism_steps(entries: dict, key_path: list[str]) -> dict[str]:
    '''
    Extracts mechanism mrv files into dict
    '''
    steps = {}
    for k, v in entries.items():
        for mech in v[key_path[0]][key_path[1]]:
            for step in mech[key_path[2]]:
                step_id = f"{k}_{mech['mechanism_id']}_{step['step_id']}"
                mrv = fetch_mrv(f"http://{step[key_path[3]]}")
                if mrv:
                    steps[step_id] = mrv
                else:
                    print(f"Failed to fetch MRV for {step_id}")
    
    return steps

def extract_mols(entries: dict, key_path: list[str]) -> dict[str]:
    '''
    Extracts mol files into dict
    '''
    mols = {}
    for k, v in entries.items():
        for cpd in v[key_path[0]][key_path[1]]:
            mol_id = f"{cpd['chebi_id']}"
            mol = fetch_mol(f"http://{cpd[key_path[2]]}")
            if mol:
                mols[mol_id] = mol
            else:
                print(f"Failed to fetch MOL for {mol_id}")
    return mols

cp = str(Path(__file__).parent / "conf")
@hydra.main(version_base=None, config_path=cp, config_name="mcsa_pull")
def main(cfg: DictConfig):
    batch_num = 0
    next_url = cfg.url
    while next_url:
        print(f"Fetching: {next_url}")
        response = fetch_json(next_url)
        entries = extract_entries(results=response['results'], key=cfg.key, fields=cfg.fields)
        print(f"Extracting mol_files from {len(entries)} entries")
        mols = extract_mols(entries=entries, key_path=cfg.mol_path)
        print(f"Extracting mechanism steps from {len(entries)} entries")
        mech_steps = extract_mechanism_steps(entries=entries, key_path=cfg.mech_step_path)

        for k, v in mech_steps.items():
            with open(Path(cfg.filepaths.raw_data) / "mcsa" / "mech_steps" / f"{k}.mrv", "w") as f:
                f.write(v)

        for k, v in mols.items():
            with open(Path(cfg.filepaths.raw_data) / "mcsa" / "mols" / f"{k}.mol", "w") as f:
                f.write(v)

        with open(Path(cfg.filepaths.raw_data) / "mcsa" / f"entries_{batch_num}.json", "w") as f:
            json.dump(entries, f, indent=4)

        batch_num += 1
        next_url = response.get('next')

if __name__ == '__main__':
    main()