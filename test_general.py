# test_general.py
# Loads a model and runs it on test data for various combinatorial optimization problems.

import torch
from utils.parsing import parse_test_args # Your modified version
import os
import numpy as np
import time
import math
from datetime import datetime
import json # For saving args

from data.loader import construct_loaders, test_datasets
from model.models import construct_model
from model.saving import load_model
from model.training import featurize_batch # Crucial: ensure this can handle single Data objects
from problem.problems import get_problem
from problem.baselines import random_hyperplane_projector
from torch_geometric.data import Data # For type checking and cloning

# --- Helper Functions (get_optimal_value, calculate_ratio) ---
def get_optimal_value(args, example, problem):
    problem_type = getattr(args, 'problem_type', 'unknown')
    num_nodes_on_example = getattr(example, 'num_nodes', None)
    if torch.is_tensor(num_nodes_on_example):
        num_nodes_on_example = num_nodes_on_example.item()

    optimal_attr = getattr(example, 'optimal', None)
    if torch.is_tensor(optimal_attr):
        optimal_attr = optimal_attr.item()
    
    y_attr = getattr(example, 'y', None)
    if torch.is_tensor(y_attr):
        y_attr = y_attr.item()

    if problem_type == 'vertex_cover':
        if optimal_attr is not None: # In ForcedRB, example.optimal is alpha(G)
            if num_nodes_on_example is not None:
                 return num_nodes_on_example - optimal_attr
            else:
                graph_size = problem.get_num_nodes(example) if hasattr(problem, 'get_num_nodes') else None
                if graph_size is not None: return graph_size - optimal_attr
        elif y_attr is not None:
            return y_attr
    elif problem_type == 'max_cut':
        if optimal_attr is not None:
            return optimal_attr
        elif y_attr is not None:
            return y_attr
    elif problem_type == 'sat':
        if optimal_attr is not None:
            return optimal_attr
        elif y_attr is not None:
            return y_attr
    return None

def calculate_ratio(args, achieved_score, optimal_value, example, problem):
    problem_type = getattr(args, 'problem_type', 'unknown')
    if optimal_value is None or np.isnan(optimal_value):
        return float('nan')

    achieved_value = 0.0
    if problem_type == 'vertex_cover':
        achieved_value = -float(achieved_score)
        if optimal_value > 0: return achieved_value / optimal_value
        elif optimal_value == 0 and achieved_value == 0: return 1.0
        elif optimal_value == 0 and achieved_value > 0: return float('inf')
        else: return float('nan')
    elif problem_type == 'max_cut':
        achieved_value = float(achieved_score)
        if optimal_value != 0: return achieved_value / abs(optimal_value)
        elif optimal_value == 0 and achieved_value == 0: return 1.0
        elif optimal_value == 0 and achieved_value != 0: return float('inf') if achieved_value > 0 else 0.0
        else: return float('nan')
    elif problem_type == 'sat':
        achieved_value = float(achieved_score)
        if optimal_value > 0: return achieved_value / optimal_value
        elif optimal_value == 0 and achieved_value == 0: return 1.0
        elif optimal_value == 0 and achieved_value > 0: return float('inf')
        else: return float('nan')
    return float('nan')

# --- Main Evaluation Function ---
def time_and_scores_eval(args, model, test_loader, problem, cli_stop_early_flag=False):
    raw_scores_list = []
    calculated_ratios_list = []
    execution_times_list = []
    optimal_values_for_instances_list = []

    total_examples_processed = 0
    batches_processed = 0
    
    # Default batch limit if not using the CLI --stop-early-testing flag for example count
    # You can make MAX_BATCHES_FOR_QUICK_TEST an argument itself if needed
    MAX_BATCHES_FOR_QUICK_TEST = 2 

    model.eval()
    with torch.no_grad():
        for batch_idx, batch_from_loader in enumerate(test_loader):
            if not batch_from_loader: continue

            # Early stopping logic:
            # 1. If CLI --stop-early-testing is used, it's based on example count.
            # 2. Otherwise (default behavior for this modified script), stop after MAX_BATCHES_FOR_QUICK_TEST.
            if cli_stop_early_flag: # CLI flag takes precedence for its specific example count
                stop_early_test_count = getattr(args, 'stop_early_test_count', 5) # Default for --stop-early-testing
                if total_examples_processed >= stop_early_test_count:
                    print(f"Stopping early after {total_examples_processed} examples (due to --stop-early-testing flag and count {stop_early_test_count}).")
                    break # Break batch loop
            elif batches_processed >= MAX_BATCHES_FOR_QUICK_TEST: # Default quick test batch limit
                 print(f"--- Manually stopping after processing {batches_processed} batches (quick test limit was {MAX_BATCHES_FOR_QUICK_TEST}). ---")
                 break # Break batch loop

            try:
                individual_graphs = batch_from_loader.to_data_list()
            except AttributeError:
                if isinstance(batch_from_loader, list):
                    individual_graphs = batch_from_loader
                else:
                    individual_graphs = [batch_from_loader]

            print(f"Processing Batch {batch_idx + 1}/{len(test_loader)} (contains {len(individual_graphs)} graphs)...")
            
            for example_idx_in_batch, example_orig in enumerate(individual_graphs):
                # Check again if CLI stop early example count is met *within* a batch
                if cli_stop_early_flag:
                    stop_early_test_count = getattr(args, 'stop_early_test_count', 5)
                    if total_examples_processed >= stop_early_test_count:
                        # This inner break will only break the example loop. The outer batch loop
                        # condition will then catch it and break fully.
                        break 
            
                start_time_instance = time.time()
                current_best_raw_score_for_example = -math.inf
                example_orig_on_device = example_orig.to(args.device)

                for attempt_idx in range(args.num_test_attempts):
                    example_for_attempt = example_orig_on_device.clone()
                    x_in, featurized_example_state = featurize_batch(args, example_for_attempt)
                    x_out = model(x_in, featurized_example_state)
                    x_proj = random_hyperplane_projector(args, x_out, featurized_example_state, problem.score)
                    x_proj = torch.where(x_proj == 0, torch.tensor(1.0, device=x_proj.device), x_proj)
                    if hasattr(args, 'penalty') and not hasattr(featurized_example_state, 'penalty'):
                         featurized_example_state.penalty = args.penalty
                    attempt_raw_score = problem.score(args, x_proj, featurized_example_state)
                    current_best_raw_score_for_example = max(current_best_raw_score_for_example, attempt_raw_score)
                
                end_time_instance = time.time()
                time_taken_instance = end_time_instance - start_time_instance
                
                raw_scores_list.append(float(current_best_raw_score_for_example))
                execution_times_list.append(time_taken_instance)
                optimal_value = get_optimal_value(args, example_orig, problem)
                optimal_values_for_instances_list.append(optimal_value if optimal_value is not None else float('nan'))
                current_ratio = calculate_ratio(args, current_best_raw_score_for_example, optimal_value, example_orig, problem)
                calculated_ratios_list.append(current_ratio)
                
                achieved_display = float(current_best_raw_score_for_example)
                if args.problem_type == 'vertex_cover': achieved_display = -achieved_display
                num_nodes_display = example_orig.num_nodes.item() if torch.is_tensor(example_orig.num_nodes) else example_orig.num_nodes

                log_msg = (f"  Ex {total_examples_processed + 1} (Nodes: {num_nodes_display if hasattr(example_orig, 'num_nodes') else 'N/A'}): "
                           f"AchievedVal={achieved_display:.2f}, OptVal={optimal_value if optimal_value is not None else 'N/A'}")
                if not np.isnan(current_ratio): log_msg += f", Ratio={current_ratio:.4f}"
                else: log_msg += ", Ratio=N/A"
                log_msg += f", Time={time_taken_instance:.3f}s, Attempts={args.num_test_attempts}"
                print(log_msg)
                total_examples_processed += 1
            
            batches_processed += 1 # Increment after processing all examples in the batch

        return raw_scores_list, calculated_ratios_list, execution_times_list, optimal_values_for_instances_list

# --- Main Script Execution ---
if __name__ == '__main__':
    args = parse_test_args()
    args_dict = vars(args)
    print("Effective Arguments for Testing (from params.txt and CLI overrides):")
    for arg_name, value in sorted(args_dict.items()):
        if isinstance(value, list) and len(value) > 10: print(f"  {arg_name}: list of length {len(value)}")
        else: print(f"  {arg_name}: {value}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available() and args.device.type == 'cuda':
        torch.cuda.manual_seed_all(args.seed)

    print(f"Loading problem definition for: {args.problem_type}")
    problem = get_problem(args)

    print(f"Constructing test loader for dataset: {args.dataset} (gen_n: {getattr(args, 'gen_n', 'N/A')})...")
    if args.dataset not in test_datasets:
        if args.use_val_set:
            print("Using validation set for testing (generated data).")
            _, test_loader, _ = construct_loaders(args)
        else:
            print("Using test set for testing (generated data).")
            _, _, test_loader = construct_loaders(args)
    else:
        print(f"Using predefined test dataset: {args.dataset}")
        mode_to_load = "val" if args.use_val_set else "test"
        if args.use_val_set: print("Using validation set for testing (predefined data).")
        else: print("Using test set for testing (predefined data).")
        test_loader = construct_loaders(args, mode=mode_to_load)

    if not test_loader:
        print("Error: Test loader could not be constructed. Exiting."); exit(1)
    
    try: num_test_examples = len(test_loader.dataset)
    except TypeError: num_test_examples = "Unknown (IterableDataset)"
    print(f"Test loader constructed. Estimated number of examples in test split: {num_test_examples}")
    if num_test_examples == 0: print("Warning: Test loader is empty.")
    
    print("Constructing model...")
    model, _ = construct_model(args)
    model_path = os.path.join(args.model_folder, args.model_file)
    print(f"Loading model weights from: {model_path}")
    if not os.path.exists(model_path): print(f"Error: Model file '{model_path}' not found. Exiting."); exit(1)
    model = load_model(model, model_path)
    model.to(args.device)
    print(f"Model '{args.model_type}' loaded and moved to {args.device}.")

    # Get the stop_early_testing flag from args (defaulted to False in parse_test_args if not present)
    cli_stop_early_flag_value = getattr(args, 'stop_early_testing', False)
    
    if args.device.type == 'cuda' and not cli_stop_early_flag_value and num_test_examples != 0 and num_test_examples != "Unknown (IterableDataset)":
        print("Running initial CUDA synchronization pass (1 example)...")
        class TempArgsSync: pass
        temp_args_sync = TempArgsSync(); setattr(temp_args_sync, 'stop_early_test_count', 1)
        for k,v in args_dict.items(): setattr(temp_args_sync, k, v) # Copy all args
        try:
            first_batch_for_sync = next(iter(test_loader))
            sync_loader = [first_batch_for_sync]
            _ = time_and_scores_eval(temp_args_sync, model, sync_loader, problem, cli_stop_early_flag=True) # Pass True here
            print("CUDA sync pass complete.")
        except Exception as e: print(f"Could not run CUDA sync pass: {e}. Proceeding.")

    print(f"Starting evaluation with {args.num_test_attempts} attempts per instance...")
    all_raw_scores, all_ratios, all_times, all_optimal_vals = time_and_scores_eval(
        args, model, test_loader, problem, cli_stop_early_flag=cli_stop_early_flag_value
    )

    if not all_raw_scores:
        print("No results to report. Evaluation might have been stopped early or test_loader was empty.")
    else:
        print(f"\n--- Final Results ({args.problem_type} on {args.dataset}) ---")
        print(f"Number of test instances evaluated: {len(all_raw_scores)}")
        mean_raw_score = np.mean(all_raw_scores); std_raw_score = np.std(all_raw_scores)
        print(f"Raw Scores: Mean={mean_raw_score:.4f}, StdDev={std_raw_score:.4f}")
        valid_ratios = [r for r in all_ratios if not np.isnan(r) and not np.isinf(r)]
        if valid_ratios:
            mean_ratio = np.mean(valid_ratios); std_ratio = np.std(valid_ratios)
            print(f"Approximation Ratios (on {len(valid_ratios)} instances with valid finite optimals/ratios): Mean={mean_ratio:.4f}, StdDev={std_ratio:.4f}")
        else: print("Approximation Ratios: Not available or all were NaN/Inf.")
        total_time = sum(all_times); avg_time_per_instance = np.mean(all_times) if all_times else 0
        print(f"Total testing time (model inference & scoring): {total_time:.2f}s")
        print(f"Average time per instance: {avg_time_per_instance:.3f}s")

        results_dict_to_save = {
            'args': {k: (str(v) if isinstance(v, torch.device) else v) for k, v in args_dict.items()},
            'raw_scores': np.array(all_raw_scores).tolist(),
            'calculated_ratios': np.array(all_ratios).tolist(),
            'optimal_values': np.array(all_optimal_vals).tolist(),
            'execution_times_per_instance': np.array(all_times).tolist(),
            'summary_stats': {
                'mean_raw_score': mean_raw_score, 'std_raw_score': std_raw_score,
                'mean_ratio (valid)': np.mean(valid_ratios) if valid_ratios else None,
                'std_ratio (valid)': np.std(valid_ratios) if valid_ratios else None,
                'num_valid_ratios': len(valid_ratios), 'total_time_sec': total_time,
                'avg_time_per_instance_sec': avg_time_per_instance,
                'num_instances_evaluated': len(all_raw_scores)
            }
        }
        test_prefix_str = args.test_prefix if args.test_prefix else f"{args.dataset}_eval"
        datetime_now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename_json = os.path.join(
            args.model_folder, 
            f"{test_prefix_str}@@{args.problem_type.upper()}_results__{datetime_now_str}.json"
        )
        with open(output_filename_json, 'w') as f: json.dump(results_dict_to_save, f, indent=4)
        print(f"Results (JSON) saved to: {output_filename_json}")
    print("Finished testing script.")