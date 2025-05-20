import copy
from itertools import product
from train import main
from time import time


# Models to test in ablation study
MODELS = ["LiftMP", "GAT", "GCNN", "GIN", "GatedGCNN"]

SPECS = {
    "vertex_cover": {
        # Synthetic Graphs
        "BA 50 100":    {"ranks": [16], "layers": [8],  "pos_enc": ["random_walk"], "steps": [20000], "valid_freq": [1000]},
        "BA 100 200":   {"ranks": [16], "layers": [8],  "pos_enc": ["random_walk"], "steps": [20000], "valid_freq": [1000]},
        "BA 400 500":   {"ranks": [16], "layers": [8],  "pos_enc": ["random_walk"], "steps": [20000], "valid_freq": [1000]},
        "ER 50 100":    {"ranks": [4],  "layers": [8],  "pos_enc": ["random_walk"], "steps": [20000], "valid_freq": [1000]},
        "ER 100 200":   {"ranks": [4],  "layers": [8],  "pos_enc": ["random_walk"], "steps": [20000], "valid_freq": [1000]},
        "ER 400 500":   {"ranks": [4],  "layers": [8],  "pos_enc": ["random_walk"], "steps": [20000], "valid_freq": [1000]},
        "HK 50 100":    {"ranks": [32], "layers": [16], "pos_enc": ["random_walk"], "steps": [20000], "valid_freq": [1000]},
        "HK 100 200":   {"ranks": [32], "layers": [16], "pos_enc": ["random_walk"], "steps": [20000], "valid_freq": [1000]},
        "HK 400 500":   {"ranks": [32], "layers": [16], "pos_enc": ["random_walk"], "steps": [20000], "valid_freq": [1000]},
        "WS 50 100":    {"ranks": [32], "layers": [16], "pos_enc": ["random_walk"], "steps": [20000], "valid_freq": [1000]},
        "WS 100 200":   {"ranks": [32], "layers": [16], "pos_enc": ["random_walk"], "steps": [20000], "valid_freq": [1000]},
        "WS 400 500":   {"ranks": [32], "layers": [16], "pos_enc": ["random_walk"], "steps": [20000], "valid_freq": [1000]},
    },
    "max_cut": {
        # Synthetic Graphs
        "BA 50 100":    {"ranks": [16], "layers": [16], "pos_enc": ["random_walk"], "steps": [20000], "valid_freq": [1000]},
        "BA 100 200":   {"ranks": [16], "layers": [16], "pos_enc": ["random_walk"], "steps": [20000], "valid_freq": [1000]},
        "BA 400 500":   {"ranks": [16], "layers": [16], "pos_enc": ["random_walk"], "steps": [20000], "valid_freq": [1000]},
        "ER 50 100":    {"ranks": [16], "layers": [16], "pos_enc": ["random_walk"], "steps": [20000], "valid_freq": [1000]},
        "ER 100 200":   {"ranks": [16], "layers": [16], "pos_enc": ["random_walk"], "steps": [20000], "valid_freq": [1000]},
        "ER 400 500":   {"ranks": [16], "layers": [16], "pos_enc": ["random_walk"], "steps": [20000], "valid_freq": [1000]},
        "HK 50 100":    {"ranks": [16], "layers": [16], "pos_enc": ["random_walk"], "steps": [20000], "valid_freq": [1000]},
        "HK 100 200":   {"ranks": [16], "layers": [16], "pos_enc": ["random_walk"], "steps": [20000], "valid_freq": [1000]},
        "HK 400 500":   {"ranks": [16], "layers": [16], "pos_enc": ["random_walk"], "steps": [20000], "valid_freq": [1000]},
        "WS 50 100":    {"ranks": [16], "layers": [8],  "pos_enc": ["random_walk"], "steps": [20000], "valid_freq": [1000]},
        "WS 100 200":   {"ranks": [16], "layers": [8],  "pos_enc": ["random_walk"], "steps": [20000], "valid_freq": [1000]},
        "WS 400 500":   {"ranks": [16], "layers": [8],  "pos_enc": ["random_walk"], "steps": [20000], "valid_freq": [1000]},
    },
}

BASE_OVERRIDES = {
    'stepwise': True,
    'steps': None,
    'valid_freq': 2000,
    'dropout': 0,
    'prefix': None,
    'model_type': None,  # Will be set during parameter generation
    'dataset': None,
    'parallel': 20,
    'infinite': True,
    'pe_dimension': None,
    'positional_encoding': None,
    'pe_dimension': None,
    'num_layers': None,
    'rank': None,
    'problem_type': None,
    'batch_size': 32,
    'lr': 0.001,
    'seed': 42,
    'hidden_channels': 32,  # Added for GNN architectures
}

DATASET_MAP = {
    'ER': 'ErdosRenyi',
    'BA': 'BarabasiAlbert',
    'WS': 'WattsStrogatz', 
    'HK': 'PowerlawCluster'
}

def set_dict_params(param_dict, model_type, num_layers, rank, pos_enc, steps, valid_freq, problem):
    param_dict.update({
        "model_type": model_type,
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
            for model_type in MODELS:
                for rank, layers, pe_enc, steps, val_freq in product(spec_params['ranks'], spec_params['layers'], spec_params['pos_enc'], spec_params['steps'], spec_params['valid_freq']):
                    dt_short_name = sep_g_params[0].strip()
                    overrides = set_dict_params(copy.deepcopy(BASE_OVERRIDES), model_type, layers, rank, pe_enc, steps, val_freq, problem)
                    
                    # Only handling graph datasets as we've removed SAT
                    gen_n = [int(sep_g_params[1].strip()), int(sep_g_params[2].strip())]
                    overrides['prefix'] = f'{problem}{model_type}{dt_short_name}{gen_n[0]}{gen_n[1]}'
                    overrides['gen_n'] = gen_n
                    overrides['dataset'] = DATASET_MAP[dt_short_name]
                    # Adjust batch size for larger graphs
                    if gen_n[0] >= 400 or gen_n[1] >= 400:
                        overrides['batch_size'] = 16
                    
                    list_params.append(overrides)
    return list_params


def train_single(overrides, id):
    start = time()
    try:
        main(overrides)
        print(f"{id}: Training Completed in {time() - start:.2f}s.")
        with open("success.log", "a") as f:
            f.write(f'{id}|{overrides}\n')
        return 1
    except Exception as e:
        print(f"{id}: Training failed in {time() - start:.2f}s with error: {e}")
        with open("recovery.log", "a") as f:
            f.write(f'{id}|{overrides}|{e}\n')
        return 0


def fetch_idxs(log_path: str = './success.log'):
    try:
        with open(log_path, "r") as f:
            done = f.readlines()
        done = [get_log_idx(x) for x in done]
        return done
    except FileNotFoundError:
        return []


def train_all(list_params):
    done_idxs = fetch_idxs('./success.log')
    failed_idxs = fetch_idxs('./recovery.log')
    to_ignore = set(done_idxs + failed_idxs)
    num_params = len(list_params)

    successes = 0
    total_start = time()

    for idx, params in enumerate(list_params, 1):
        if idx in to_ignore:
            print(f"Skipping {idx}, already done.")
            continue
        successes += train_single(params, f'[{idx}/{num_params}]')
    
    print(f"Total training time: {time() - total_start:.2f}s.")
    print(f"Successfully completed {successes}/{num_params - len(to_ignore)} experiments.")


def get_log_idx(line_str):
    return int(line_str.split("|")[0].split('/')[0].replace('[','').strip())


def run_recovery():
    with open("recovery.log", "r") as f:
        failed = f.readlines()
    failed_idxs = [get_log_idx(x) for x in failed]
    list_params = generate_parameters()
    tot_size = copy.deepcopy(len(list_params))
    list_params = [x for idx, x in enumerate(list_params, 1) if idx in failed_idxs]

    for idx, params in zip(failed_idxs, list_params):
        train_single(params, f'[{idx}/{tot_size}]')


if __name__ == "__main__":
    # Generate all experiment parameters for model ablations
    list_params = generate_parameters()
    
    # Print experiment summary
    model_counts = {}
    dataset_counts = {}
    problem_counts = {}
    
    for params in list_params:
        model = params.get('model_type')
        dataset = params.get('dataset')
        problem = params.get('problem_type')
        
        model_counts[model] = model_counts.get(model, 0) + 1
        dataset_counts[dataset] = dataset_counts.get(dataset, 0) + 1
        problem_counts[problem] = problem_counts.get(problem, 0) + 1
    
    print("=== Experiment Summary ===")
    print(f"Total experiments: {len(list_params)}")
    print(f"Models: {model_counts}")
    print(f"Datasets: {dataset_counts}")
    print(f"Problems: {problem_counts}")
    print("=========================")
    
    # Run all experiments
    train_all(list_params)

    # Uncomment the line below to run recovery on failed runs
    # run_recovery()

    # On current setup:
    # - HK, WS, BA, ER take about ~1329.55 seconds to train each model'''