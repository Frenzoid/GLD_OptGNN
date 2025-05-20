import json
import os
import torch
from argparse import ArgumentParser, _StoreTrueAction, _StoreFalseAction # Import specific action types

# Assuming these imports are correct and files exist
from data.loader import construct_loaders
from problem.problems import get_problem
from model.models import construct_model
from utils.parsing import add_general_args, add_train_args, add_dataset_args, modify_train_args, check_args
from model.training import train, validate


def add_artifact_dir_arg(parser: ArgumentParser):
    """Adds argument for specifying the run artifact directory."""
    group = parser.add_argument_group('artifact_args')
    group.add_argument('--run-artifact-dir', type=str, default=None, dest='run_artifact_dir',
                       help='Directory to save run-specific artifacts like models and params.txt. '
                            'If None, defaults might be based on log_dir or current dir.')

def main(overrides):
    """
    Main training function, callable with a dictionary of overrides.
    """
    parser = ArgumentParser()
    add_general_args(parser)
    add_train_args(parser)
    add_dataset_args(parser)
    add_artifact_dir_arg(parser) # Ensure this is called

    cli_args = []
    for key, value in overrides.items():
        cli_key = f"--{key.replace('_', '-')}"
        action_obj = None
        for act in parser._actions:
            if act.dest == key:
                action_obj = act
                break
            if cli_key in act.option_strings:
                action_obj = act
                break
        
        if action_obj is None:
            print(f"    WARNING: No action_obj found for override key '{key}' (cli_key: '{cli_key}'). If this key is a direct arg, check parser setup. Skipping this override for cli_args construction.")
            continue

        is_store_true_false_action = isinstance(action_obj, (_StoreTrueAction, _StoreFalseAction))
        
        if is_store_true_false_action:
            if isinstance(action_obj, _StoreTrueAction) and value is True:
                cli_args.append(cli_key)
            elif isinstance(action_obj, _StoreFalseAction) and value is False:
                cli_args.append(cli_key)
        elif value is not None:
            cli_args.append(cli_key)
            if isinstance(value, list):
                cli_args.extend(map(str, value))
            else:
                cli_args.append(str(value))

    print("Constructed cli_args by train.main:", cli_args)
    
    try:
        args = parser.parse_args(cli_args)
    except SystemExit as e:
        print(f"ERROR: Argument parsing failed in train.main with cli_args: {cli_args}")
        print(f"SystemExit: {e}")
        raise

    torch.manual_seed(args.seed if isinstance(args.seed, int) else int(args.seed))

    # Call these after parsing cli_args within this function
    modify_train_args(args) # This will set args.log_dir based on args.prefix etc. and also args.device
    check_args(args)

    # Determine the primary output directory
    # args.run_artifact_dir comes from TUdebug.py specifically for this callable main.
    # args.log_dir is set by modify_train_args (based on prefix or timestamp).
    # We want to prioritize run_artifact_dir if provided by the orchestrator.
    output_directory = args.run_artifact_dir if args.run_artifact_dir else args.log_dir
    if not output_directory: # Should be extremely rare now
        print("WARNING: Neither run_artifact_dir nor log_dir is effectively set. Defaulting output_directory to './default_train_output_final'")
        output_directory = "./default_train_output_final"
        # If we defaulted output_directory, ensure args.log_dir reflects it if it was the one missing,
        # so that if model.training.train internally uses args.log_dir, it's consistent.
        if not args.log_dir:
            args.log_dir = output_directory
    
    os.makedirs(output_directory, exist_ok=True)
    print(f"All artifacts for this run will be saved in: {output_directory}")

    # Save parameters to the determined output_directory
    args_dict = vars(args)
    if hasattr(args, 'device') and not isinstance(args.device, str):
        args_dict['device'] = str(args.device)
    
    params_file_path = os.path.join(output_directory, 'params.txt')
    try:
        with open(params_file_path, 'w') as f:
            json.dump(args_dict, f, indent=4)
        print(f"Saved parameters to {params_file_path}")
    except Exception as e:
        print(f"Error saving params.txt: {e}")

    train_loader, val_loader, test_loader = construct_loaders(args)
    model, optimizer = construct_model(args)
    problem = get_problem(args)
    
    # Ensure train function can use output_directory for saving checkpoints
    # MODIFICATION: Removed output_dir=output_directory
    # Assumes model.training.train will use args.log_dir or args.run_artifact_dir if needed
    train(args, model, train_loader, optimizer, problem, val_loader=val_loader, test_loader=test_loader)
    
    valid_loss = validate(args, model, val_loader, problem)
    
    # Save done.txt to the determined output_directory
    done_file_path = os.path.join(output_directory, 'done.txt')
    try:
        with open(done_file_path, 'w') as file:
            file.write(f"Validation Loss: {valid_loss}\n")
            file.write("done.\n")
        print(f"Wrote done.txt to {done_file_path}")
    except Exception as e:
        print(f"Error writing done.txt: {e}")


if __name__ == '__main__':
    from utils.parsing import parse_train_args

    args_main = parse_train_args()
    print("Running train.py directly using its __main__ block.")
    print("Parsed args in __main__ (using parse_train_args):", args_main)
    
    if hasattr(args_main, 'seed') and not isinstance(args_main.seed, int):
        args_main.seed = int(args_main.seed)
    torch.manual_seed(args_main.seed)

    # args_main.log_dir is set by modify_train_args called within parse_train_args
    os.makedirs(args_main.log_dir, exist_ok=True) # This is the primary output dir for standalone run

    args_main_dict = vars(args_main)
    if hasattr(args_main, 'device') and not isinstance(args_main.device, str):
        args_main_dict['device'] = str(args_main.device)

    json.dump(args_main_dict, open(os.path.join(args_main.log_dir, 'params.txt'), 'w'), indent=4)

    train_loader, val_loader, test_loader = construct_loaders(args_main)
    model, optimizer = construct_model(args_main)
    problem = get_problem(args_main)
    
    # MODIFICATION: Removed output_dir=args_main.log_dir
    # Assumes model.training.train will use args_main.log_dir if needed
    train(args_main, model, train_loader, optimizer, problem, val_loader=val_loader, test_loader=test_loader)
    
    valid_loss = validate(args_main, model, val_loader, problem)
    
    with open(os.path.join(args_main.log_dir, 'done.txt'), 'w') as file:
        file.write(f"Validation Loss: {valid_loss}\n")
        file.write("done.\n")