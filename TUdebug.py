#!/usr/bin/env python3
"""
Fixed version of TUDebugging.py with improved error handling and directory management
"""

import copy
import os
import sys
import traceback
from itertools import product
from time import time
import logging

# Attempt to import the main function from train.py
try:
    from train import main
except ImportError:
    print("ERROR: Could not import 'main' from train.py")
    print("Make sure train.py is in the current directory or in PYTHONPATH")
    sys.exit(1)

# --- Define a base directory for debug outputs ---
DEBUG_BASE_DIR = "debug_session_outputs"

# Setup logging
def setup_logging(prefix, run_id_str):
    """Set up logging to both console and a file within the debug directory."""
    # Use a more specific log directory per run
    sanitized_run_id = run_id_str.replace('[','').replace(']','').replace('/','_')
    log_dir = os.path.join(DEBUG_BASE_DIR, "run_logs", f"{prefix}_{sanitized_run_id}")
    os.makedirs(log_dir, exist_ok=True)

    log_file = os.path.join(log_dir, "output.txt")

    # Get root logger
    logger = logging.getLogger()
    # Remove all existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return log_file

# Simplified SPECS with just one TU small dataset
SPECS = {
    "vertex_cover": {
        "MUTAG": {"ranks": [16], "layers": [8], "pos_enc": ["random_walk"], "steps": [100000], "valid_freq": [1000]},
    },
    "max_cut": {
        "MUTAG": {"ranks": [4], "layers": [8], "pos_enc": ["random_walk"], "steps": [100000], "valid_freq": [1000]},
    },
}

BASE_OVERRIDES = {
    'stepwise': True,
    'steps': None,
    'valid_freq': 2000,
    'dropout': 0,
    'prefix': None,
    'model_type': 'LiftMP',
    'dataset': None,
    'parallel': 20,
    'infinite': True,
    'pe_dimension': None,
    'positional_encoding': None,
    'num_layers': None,
    'rank': None,
    'problem_type': None,
    'batch_size': 32,
    'penalty': 0.003,
    'lr': 0.001,
    'seed': 42,
    # Add a placeholder for the run-specific output directory
    'run_artifact_dir': None,
}

def set_dict_params(param_dict, num_layers, rank, pos_enc, steps, valid_freq, problem):
    param_dict.update({
        "num_layers": num_layers,
        "rank": rank,
        "positional_encoding": pos_enc,
        "steps": steps,
        "valid_freq": valid_freq,
        "problem_type": problem,
        'pe_dimension': 8 if rank == 32 else int(rank / 2.0),
    })
    return param_dict

def generate_parameters():
    list_params = []
    for problem, data in SPECS.items():
        for general_params, spec_params in data.items():
            sep_g_params = general_params.split()
            for rank, layers, pe_enc, steps, val_freq in product(spec_params['ranks'], spec_params['layers'], spec_params['pos_enc'], spec_params['steps'], spec_params['valid_freq']):
                dt_short_name = sep_g_params[0].strip()
                overrides = set_dict_params(copy.deepcopy(BASE_OVERRIDES), layers, rank, pe_enc, steps, val_freq, problem)
                overrides['prefix'] = f'{problem}_{dt_short_name}'
                overrides['dataset'] = dt_short_name
                list_params.append(overrides)
    return list_params

def train_single(overrides, id_str):
    # Define and create run-specific artifact directory
    sanitized_run_id = id_str.replace('[','').replace(']','').replace('/','_')
    run_artifact_dir = os.path.join(DEBUG_BASE_DIR, "model_outputs", f"{overrides['prefix']}_{sanitized_run_id}")
    
    # Ensure the directory exists before passing it to train.main
    os.makedirs(run_artifact_dir, exist_ok=True)
    overrides['run_artifact_dir'] = run_artifact_dir # Pass this to train.main
    
    # Set up logging for this specific training run
    log_file = setup_logging(overrides['prefix'], id_str)
    logging.info(f"Starting training for {overrides['prefix']} (Run ID: {id_str})")
    logging.info(f"Parameters: {overrides}")
    logging.info(f"Run artifacts will be saved to: {run_artifact_dir}")
    logging.info(f"Detailed logs in: {log_file}")

    # Create a marker file to verify the directory exists and is writable
    try:
        with open(os.path.join(run_artifact_dir, ".dir_check"), "w") as f:
            f.write("Directory check - this file verifies the directory exists and is writable")
        logging.info(f"Directory check passed - {run_artifact_dir} is writable")
    except Exception as e:
        logging.error(f"Could not write to directory {run_artifact_dir}: {e}")

    start = time()
    try:
        logging.info(f"Calling train.main with run_artifact_dir={run_artifact_dir}")
        
        # Print details of the train module to help diagnose import issues
        logging.info(f"train.main function: {main}")
        logging.info(f"train module path: {sys.modules.get('train', None).__file__ if 'train' in sys.modules else 'Not found'}")
        
        # Call train.main with the overrides
        main(overrides) 
        
        end_time = time() - start
        logging.info(f"{id_str}: Training Completed in {end_time:.2f}s.")
        
        # Check if any files were created in the run_artifact_dir
        files = os.listdir(run_artifact_dir)
        file_count = len([f for f in files if f != ".dir_check"])
        if file_count > 0:
            logging.info(f"Success! {file_count} files created in {run_artifact_dir}: {files}")
        else:
            logging.warning(f"Warning: No output files created in {run_artifact_dir} (except .dir_check)")
        
        # Log success
        with open(os.path.join(DEBUG_BASE_DIR, "success.log"), "a") as f:
            f.write(f'{id_str}|{overrides}\n')
        return 1
    except Exception as e:
        end_time = time() - start
        logging.error(f"{id_str}: Training failed in {end_time:.2f}s with error: {e}")
        logging.error(f"Traceback: {traceback.format_exc()}")  # Print full traceback
        with open(os.path.join(DEBUG_BASE_DIR, "recovery.log"), "a") as f:
            f.write(f'{id_str}|{overrides}|{e}\n')
        return 0

def fetch_idxs(log_file_name: str):
    log_path = os.path.join(DEBUG_BASE_DIR, log_file_name)
    try:
        with open(log_path, "r") as f:
            done = f.readlines()
        done = [get_log_idx(x) for x in done]
        return done
    except FileNotFoundError:
        return []

def get_log_idx(line_str):
    return int(line_str.split("|")[0].split('/')[0].replace('[','').strip())

def train_all(list_params):
    # Ensure base debug directory and its subdirectories exist
    os.makedirs(DEBUG_BASE_DIR, exist_ok=True)
    os.makedirs(os.path.join(DEBUG_BASE_DIR, "run_logs"), exist_ok=True)
    os.makedirs(os.path.join(DEBUG_BASE_DIR, "model_outputs"), exist_ok=True)

    done_idxs = fetch_idxs('success.log')
    failed_idxs = fetch_idxs('recovery.log')
    to_ignore = set(done_idxs + failed_idxs)
    num_params = len(list_params)

    # This initial logging goes to the main_process.txt in DEBUG_BASE_DIR
    logging.info(f"Starting training for {num_params} parameter configurations")
    logging.info(f"Output directory: {os.path.abspath(DEBUG_BASE_DIR)}")
    logging.info(f"Done indices: {done_idxs}")
    logging.info(f"Failed indices: {failed_idxs}")

    successes = 0
    total_start = time()

    for idx, params in enumerate(list_params, 1):
        run_id_str = f'[{idx}/{num_params}]'
        if idx in to_ignore:
            logging.info(f"Skipping run {run_id_str} ({params['prefix']}), already processed.")
            continue
        
        logging.info(f"Starting run {run_id_str} for {params['problem_type']} on {params['dataset']}")
        successes += train_single(copy.deepcopy(params), run_id_str)

    total_time = time() - total_start
    
    # Restore global logging for final summary
    logger = logging.getLogger()
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(DEBUG_BASE_DIR, "main_debug_process.txt")),
            logging.StreamHandler()
        ]
    )
    logging.info(f"Total training time: {total_time:.2f}s.")
    logging.info(f"Successful runs in this session: {successes}/{num_params - len(to_ignore)}")
    logging.info(f"Total runs skipped: {len(to_ignore)}")


if __name__ == "__main__":
    # Ensure base debug directory exists for the main log file
    os.makedirs(DEBUG_BASE_DIR, exist_ok=True)

    # Setup global logging (for the main process)
    main_log_file_path = os.path.join(DEBUG_BASE_DIR, "main_debug_process.txt")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(main_log_file_path),
            logging.StreamHandler()
        ]
    )

    logging.info("Debug script started")
    logging.info(f"All outputs will be under: {os.path.abspath(DEBUG_BASE_DIR)}")
    logging.info(f"Main process log: {main_log_file_path}")
    
    # Log system information
    logging.info(f"Python version: {sys.version}")
    logging.info(f"Current working directory: {os.getcwd()}")
    logging.info(f"Available train modules: {'train' in sys.modules}")

    # For a fresh start, uncomment these lines:
    # for f in ["success.log", "recovery.log"]:
    #     p = os.path.join(DEBUG_BASE_DIR, f)
    #     if os.path.exists(p):
    #         os.remove(p)
    #         logging.info(f"Removed old {f} for a fresh debug session.")

    list_params = generate_parameters()
    logging.info(f"Generated {len(list_params)} parameter configurations")

    train_all(list_params)

    logging.info("Debug script completed")