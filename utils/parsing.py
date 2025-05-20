import math
import os
import argparse # Import the top-level argparse module
from argparse import ArgumentParser, Namespace, Action # Import specific classes
import numpy as np
import torch
from datetime import datetime
import json
import hashlib # For hash_dict

# --- Helper Functions ---

def hash_dict(d: dict) -> str:
    """Converts a dictionary to a reproducible hash."""
    # Convert the dictionary to a sorted tuple of key-value pairs to ensure order
    sorted_items = str(tuple(sorted(d.items())))
    # Hash the string representation
    hash_value = hashlib.sha256(sorted_items.encode()).hexdigest()
    return hash_value

def read_params_from_folder(model_folder_path: str) -> dict:
    """Reads a params.txt file from a given folder and returns it as a dictionary."""
    params_file = os.path.join(model_folder_path, 'params.txt')
    if not os.path.exists(params_file):
        raise FileNotFoundError(f"params.txt not found in {model_folder_path}")
    with open(params_file, 'r') as f:
        params = json.load(f)
    return params

def check_args(args: Namespace):
    """Validates parsed arguments."""
    if hasattr(args, 'problem_type') and args.problem_type == 'sat': # Check if problem_type exists
        if hasattr(args, 'dataset') and args.dataset != 'random-sat': # Check if dataset exists
            raise ValueError(f"dataset = {args.dataset} not valid for problem_type = {args.problem_type}")
    return

# Custom argparse action for enforcing variable number of args for an option
def required_length(nmin: int, nmax: int):
    class RequiredLength(Action): # Inherit from argparse.Action
        def __call__(self, parser, args, values, option_string=None):
            if not nmin <= len(values) <= nmax:
                msg = f'argument "{self.dest}" requires between {nmin} and {nmax} arguments, got {len(values)}'
                raise argparse.ArgumentTypeError(msg)
            setattr(args, self.dest, values)
    return RequiredLength

# --- Argument Grouping Functions ---

def add_general_args(parser: ArgumentParser):
    group = parser.add_argument_group('General Arguments')
    group.add_argument('--problem-type', type=str, default='max_cut', dest='problem_type',
                       choices=['max_cut', 'vertex_cover', 'max_clique', 'sat'],
                       help='Problem to solve.')
    group.add_argument('--seed', type=int, default=0, 
                       help='PyTorch random seed.')
    group.add_argument('--prefix', type=str, default=None,
                       help='Optional prefix for run output folder name (defaults to timestamp).')

def add_dataset_args(parser: ArgumentParser):
    group = parser.add_argument_group('Dataset Arguments')
    group.add_argument('--dataset', type=str, default='ErdosRenyi',
                       choices=[
                           'ErdosRenyi', 'BarabasiAlbert', 'PowerlawCluster', 'WattsStrogatz',
                           'ForcedRB', 'ENZYMES', 'PROTEINS', 'IMDB-BINARY', 'MUTAG',
                           'COLLAB', 'REDDIT-MULTI-5K', 'REDDIT-MULTI-12K', 'REDDIT-BINARY',
                           'random-sat', 'kamis', 'gset',
                       ],
                       help='Dataset to use.')
    group.add_argument('--data-seed', type=int, default=0, dest='data_seed',
                       help='Seed for generated datasets.')
    group.add_argument('--parallel', type=int, default=0,
                       help='Number of parallel workers for data generation.')
    group.add_argument('--num-graphs', type=int, default=1000, dest='num_graphs',
                       help='Number of graphs to generate (if not infinite).')
    group.add_argument('--infinite', action='store_true',
                       help='Enable infinite generation for generated datasets (flag implies True).')
    parser.set_defaults(infinite=False)

    group.add_argument('--gen-n', nargs='+', type=int, default=None, action=required_length(1, 2), dest='gen_n',
                       help='Range for n parameter (e.g., number of vertices). E.g., "100" or "50 150".')
    group.add_argument('--gen-m', nargs='+', type=int, default=None, action=required_length(1, 2), dest='gen_m',
                       help='m parameter (meaning varies by dataset).')
    group.add_argument('--gen-k', nargs='+', type=int, default=None, action=required_length(1, 2), dest='gen_k',
                       help='k parameter (meaning varies by dataset).')
    group.add_argument('--gen-p', nargs='+', type=float, default=None, action=required_length(1, 2), dest='gen_p',
                       help='p parameter (meaning varies by dataset).')

    group.add_argument('--positional-encoding', type=str, default=None, dest='positional_encoding',
                       choices=['laplacian_eigenvector', 'random_walk'],
                       help='Type of positional encoding to use.')
    group.add_argument('--pe-dimension', type=int, default=8, dest='pe_dimension',
                       help='Dimension of the positional encoding.')
    group.add_argument('--split-seed', type=int, default=0, dest='split_seed',
                       help='Seed for train/val/test data split.')

def add_train_args(parser: ArgumentParser):
    group = parser.add_argument_group('Training Arguments')
    group.add_argument('--model-type', type=str, default='LiftMP', dest='model_type',
                       choices=['LiftMP', 'FullMP', 'GIN', 'GAT', 'GCNN', 'GatedGCNN', 'NegationGAT', 'ProjectMP', 'Nikos'],
                       help='Type of model architecture.')
    group.add_argument('--num-layers', type=int, default=12, dest='num_layers', help='Number of GNN layers.')
    # ... (rest of add_train_args is assumed to be the same as you provided) ...
    group.add_argument('--repeat-lift-layers', nargs='+', type=int, default=None, dest='repeat_lift_layers',
                       help='List of repeat counts for each lift layer.')
    group.add_argument('--num-layers-project', type=int, default=4, dest='num_layers_project',
                       help='Number of projection layers (for FullMP).')
    group.add_argument('--rank', type=int, default=32,
                       help='Dimensionality of node vectors (solution matrix rank).')
    group.add_argument('--dropout', type=float, default=0.1,
                       help='Dropout rate.')
    group.add_argument('--hidden-channels', type=int, default=32, dest='hidden_channels',
                       help='Dimension of hidden channels in GNN layers.')
    group.add_argument('--norm', type=str, default="BatchNorm",
                       help='Type of normalization layer.')
    group.add_argument('--heads', type=int, default=4,
                       help='Number of attention heads (for GAT).')
    group.add_argument('--finetune-from', type=str, default=None, dest='finetune_from',
                       help='Path to model file for loading weights for fine-tuning.')
    group.add_argument('--lift-file', type=str, default=None, dest='lift_file',
                       help='Path to model file for loading lift network weights.')

    group.add_argument('--lr', type=float, default=0.001, help='Learning rate.')
    group.add_argument('--batch-size', type=int, default=32, dest='batch_size', help='Training batch size.')
    group.add_argument('--valid-freq', type=int, default=1000, dest='valid_freq',
                       help='Validation frequency (steps/epochs). 0 for no validation.')
    group.add_argument('--save-freq', type=int, default=1000, dest='save_freq',
                       help='Model save frequency (steps/epochs). 0 for end-of-training only.')
    group.add_argument('--penalty', type=float, default=1.0,
                       help='Penalty for constraint violations.')

    group.add_argument('--train-by-epochs', action='store_false', dest='stepwise',
                       help='Train by epochs instead of gradient steps. (Default: train by steps, i.e. stepwise=True)')
    parser.set_defaults(stepwise=True)

    group.add_argument('--steps', type=int, default=50000,
                       help='Total training steps (if stepwise).')
    group.add_argument('--epochs', type=int, default=100,
                       help='Total training epochs (if not stepwise).')
    group.add_argument('--train-fraction', type=float, default=0.8, dest='train_fraction',
                       help='Fraction of data for training.')


def modify_train_args(args: Namespace):
    if args.finetune_from is not None:
        model_folder = os.path.dirname(args.finetune_from)
        try:
            model_args_dict = read_params_from_folder(model_folder)
            print(f"Finetuning from: {args.finetune_from}. Loaded pretrain_config from params.txt.")
            pretrain_config_keys = [
                'problem_type', 'model_type', 'num_layers', 'num_layers_project',
                'rank', 'dropout', 'hidden_channels', 'norm', 'heads',
                'positional_encoding', 'pe_dimension', 'repeat_lift_layers', 'lift_file'
            ]
            for k in pretrain_config_keys:
                if k in model_args_dict: setattr(args, k, model_args_dict[k])
                else: print(f"Warning: Key '{k}' not found in pretrain_config from {model_folder}/params.txt")
        except FileNotFoundError: print(f"Warning: params.txt not found for finetuning model at {model_folder}.")
        except json.JSONDecodeError: print(f"Warning: Could not decode params.txt from {model_folder}.")

    if args.prefix is None:
        args.log_dir = os.path.join("training_runs", datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    else:
        params_to_hash = {k: v for k, v in vars(args).items() if k not in ['device', 'log_dir', 'run_artifact_dir', 'prefix', 'finetune_from', 'lift_file']}
        hashed_config = hash_dict(params_to_hash)
        args.log_dir = os.path.join("training_runs", args.prefix, f"paramhash_{hashed_config[:12]}")

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {args.device}")

def parse_train_args() -> Namespace:
    parser = ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_general_args(parser)
    add_train_args(parser)
    add_dataset_args(parser)
    args = parser.parse_args()
    modify_train_args(args) # Modifies args in place, e.g., sets log_dir, device
    check_args(args)      # Validates args
    return args

def parse_test_args() -> Namespace:
    parser = ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    test_arg_group = parser.add_argument_group('Testing Arguments') # Keep reference to the group
    test_arg_group.add_argument('--model-folder', type=str, required=True, dest='model_folder',
                       help='Folder containing the trained model and params.txt.')
    test_arg_group.add_argument('--model-file', type=str, default='best_model.pt', dest='model_file',
                       help='Name of the model state_dict file within model_folder.')
    test_arg_group.add_argument('--test-prefix', type=str, default=None, dest='test_prefix',
                       help='Optional prefix for test output files.')
    test_arg_group.add_argument('--use-val-set', action='store_true', dest='use_val_set',
                       help='Evaluate on the validation set instead of the test set.')
    parser.set_defaults(use_val_set=False)
    test_arg_group.add_argument('--test-problem-type', type=str, default=None, dest='test_problem_type',
                       help='Override problem_type from params.txt for testing (e.g. for transfer).')
    test_arg_group.add_argument('--test-seed', type=int, default=None, dest='test_seed',
                       help='Override seed from params.txt for testing.')
    test_arg_group.add_argument('--test-device', type=str, default=None, choices=['cuda', 'cpu'], dest='test_device',
                       help='Override device for testing (cuda/cpu).')
    test_arg_group.add_argument('--stop-early-testing', action='store_true', dest='stop_early_testing',
                       help='Stop evaluation early after a few examples for quick script testing.')
    parser.set_defaults(stop_early_testing=False)
    test_arg_group.add_argument('--num-test-attempts', type=int, default=100,
                       dest='num_test_attempts',
                       help='Number of random hyperplane projection attempts per instance. ')
    # Add max_test_examples if you implemented it
    # test_arg_group.add_argument('--max-test-examples', type=int, default=None, dest='max_test_examples',
    #                    help='Maximum number of examples to process from the test set.')


    dataset_arg_group = add_dataset_args(parser)
    cli_args = parser.parse_args()
    model_train_params_dict = read_params_from_folder(cli_args.model_folder)
    final_args_dict = model_train_params_dict.copy()

    # --- Type Conversions from params.txt (JSON strings to Python types) ---
    params_to_convert_to_int = [
        'seed', 'data_seed', 'parallel', 'num_graphs', 'pe_dimension', 'split_seed',
        'num_layers', 'num_layers_project', 'rank', 'hidden_channels', 'heads',
        'batch_size', 'valid_freq', 'save_freq', 'steps', 'epochs', 'num_test_attempts'
        # 'max_test_examples' # if you added this
    ]
    params_to_convert_to_float = ['dropout', 'lr', 'penalty', 'train_fraction']
    boolean_params_from_string = ['infinite', 'stepwise', 'use_val_set', 'stop_early_testing']


    for param_name in params_to_convert_to_int:
        if param_name in final_args_dict and final_args_dict[param_name] is not None:
            try:
                if isinstance(final_args_dict[param_name], list):
                    final_args_dict[param_name] = [int(x) for x in final_args_dict[param_name] if x is not None]
                else: final_args_dict[param_name] = int(final_args_dict[param_name])
            except (ValueError, TypeError) as e: print(f"Warning: Could not convert param '{param_name}' (value: {final_args_dict[param_name]}) to int: {e}.")
    for param_name in params_to_convert_to_float:
        if param_name in final_args_dict and final_args_dict[param_name] is not None:
            try:
                if isinstance(final_args_dict[param_name], list):
                    final_args_dict[param_name] = [float(x) for x in final_args_dict[param_name] if x is not None]
                else: final_args_dict[param_name] = float(final_args_dict[param_name])
            except (ValueError, TypeError) as e: print(f"Warning: Could not convert param '{param_name}' (value: {final_args_dict[param_name]}) to float: {e}.")
    for param_name in boolean_params_from_string:
        if param_name in final_args_dict and isinstance(final_args_dict[param_name], str):
            val_lower = final_args_dict[param_name].lower()
            if val_lower == 'true': final_args_dict[param_name] = True
            elif val_lower == 'false': final_args_dict[param_name] = False
    # --- End Type Conversions ---

    # Override with CLI args (CLI > params.txt (now typed) > parser default)
    final_args_dict['model_folder'] = cli_args.model_folder # Always from CLI
    
    # Iterate over all arguments defined in the parser to handle overrides systematically
    # This helps ensure all CLI args have a chance to override params.txt or set their defaults.
    for action in parser._actions:
        dest = action.dest
        if dest == 'help' or dest == 'model_folder': continue # Skip help and already handled model_folder

        cli_value = getattr(cli_args, dest)
        parser_default = parser.get_default(dest)

        # If CLI value was actively set (i.e., different from its parser default, or it's a flag that's True)
        was_cli_set = (cli_value != parser_default)
        if isinstance(action, argparse._StoreTrueAction) and cli_value: # Flag was passed
            was_cli_set = True
        if isinstance(action, argparse._StoreFalseAction) and not cli_value: # Flag was passed
             was_cli_set = True


        if was_cli_set:
            if final_args_dict.get(dest) != cli_value:
                 print(f"Overriding '{dest}' from params.txt value '{final_args_dict.get(dest)}' to CLI value '{cli_value}'.")
            final_args_dict[dest] = cli_value
        elif dest not in final_args_dict: # Not in params.txt and not set by CLI, so use parser default
            final_args_dict[dest] = parser_default
    
    # Ensure boolean flags that were not in params and not set on CLI get their correct default
    for action in parser._actions:
        dest = action.dest
        if dest not in final_args_dict: # If still not in dict after above logic
            if isinstance(action, argparse._StoreTrueAction):
                final_args_dict[dest] = False # Default for store_true is False
            elif isinstance(action, argparse._StoreFalseAction):
                final_args_dict[dest] = True  # Default for store_false is True

    # Set device (this logic now uses the potentially updated final_args_dict['device'])
    if cli_args.test_device: # CLI for device has highest precedence
        final_args_dict['device'] = torch.device(cli_args.test_device)
    elif 'device' in final_args_dict:
        device_val = final_args_dict['device']
        if isinstance(device_val, str):
            if 'cuda' in device_val and not torch.cuda.is_available():
                print(f"Warning: Device '{device_val}' from params.txt requested CUDA, but CUDA is not available. Using CPU.")
                final_args_dict['device'] = torch.device('cpu')
            else:
                final_args_dict['device'] = torch.device(device_val)
        # If it's already a torch.device object (e.g. from a previous run's params.txt), keep it
    else: # Default if not in params.txt and not in CLI
        final_args_dict['device'] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Ensure it's a torch.device object at the end
    if isinstance(final_args_dict.get('device'), str):
        final_args_dict['device'] = torch.device(final_args_dict['device'])

    print(f"Testing with device: {final_args_dict['device']}")

    final_args_ns = Namespace(**final_args_dict)
    # final_args_ns.model_folder is already set because it's in final_args_dict

    # Legacy handling
    if hasattr(final_args_ns, "vc_penalty") and not hasattr(final_args_ns, 'penalty'):
        final_args_ns.penalty = final_args_ns.vc_penalty
    if hasattr(final_args_ns, 'valid_fraction') and not hasattr(final_args_ns, 'train_fraction'):
        test_fraction = getattr(final_args_ns, 'test_fraction', 0.0)
        final_args_ns.train_fraction = 1.0 - final_args_ns.valid_fraction - test_fraction

    check_args(final_args_ns)
    return final_args_ns

def modify_baseline_args(args: Namespace):
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.prefix is None:
        args.log_dir = os.path.join("baseline_runs", datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    else:
        params_to_hash = {k: v for k, v in vars(args).items() if k not in ['device', 'log_dir', 'prefix']}
        hashed_config = hash_dict(params_to_hash)
        args.log_dir = os.path.join("baseline_runs", args.prefix, f"paramhash_{hashed_config[:12]}")
    args.batch_size = 1

def parse_baseline_args() -> Namespace:
    parser = ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    add_general_args(parser)
    add_dataset_args(parser)
    group = parser.add_argument_group('Baseline Specific Arguments')
    # ... (rest of parse_baseline_args)
    group.add_argument('--sdp', action='store_true', help='Run SDP baseline.')
    group.add_argument('--gurobi', action='store_true', help='Run Gurobi baseline.')
    group.add_argument('--gurobi-timeout', type=float, default=3600.0, dest='gurobi_timeout',
                       help='Timeout for Gurobi solver in seconds.')
    group.add_argument('--greedy', action='store_true', help='Run a greedy baseline.')
    group.add_argument('--start-index', type=int, default=None, dest='start_index',
                       help='Start index in dataset for partial evaluation.')
    group.add_argument('--end-index', type=int, default=None, dest='end-index',
                       help='End index (exclusive) in dataset for partial evaluation.')
    args = parser.parse_args()
    modify_baseline_args(args)
    check_args(args)
    return args

print(f"--- utils.parsing.py has been loaded. All functions defined. ---")
print(f"    hash_dict defined: {'hash_dict' in globals()}")
print(f"    read_params_from_folder defined: {'read_params_from_folder' in globals()}")
print(f"    check_args defined: {'check_args' in globals()}")