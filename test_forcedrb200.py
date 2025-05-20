# test_forcedrb_modified.py
# loads model and runs it on data.

import torch
from utils.parsing import parse_test_args # Assuming this correctly loads params from model_folder
import json
import os
from data.loader import construct_loaders, test_datasets # Assuming this can generate/load 1000 ForcedRB test graphs
from model.training import validate # Not directly used for testing here, but featurize_batch is
from model.models import construct_model
from model.saving import load_model
import pickle
from datetime import datetime
import numpy as np
from problem.baselines import random_hyperplane_projector
from model.training import featurize_batch # Used for preparing input
import time
from problem.problems import get_problem # Used to get problem-specific loss and score functions
import math

# from data.gset import load_gset # Not used for ForcedRB

'''
python test_forcedrb_modified.py --model_folder="/teamspace/studios/this_studio/bespoke-gnn4do/training_runs/vertex_cover_RB200/paramhash:ecbad02ec728e09d92fe4b2af25f1bf68b0d04f8451cfd61bd6ae36b1937c04c" \
    --model_file=best_model.pt --test_prefix=RB200_VC_Eval --problem_type=vertex_cover

Will load the dataset and parameters from the params.txt in the model folder.
It will also use the dataset generation parameters (gen_n, gen_k) from that params.txt
to create the ForcedRB test instances.
'''


def time_and_scores(args, model, test_loader, problem, stop_early=False):
    total_loss = 0.
    total_count = 0
    times = []
    scores_from_problem_score = [] # Stores the direct output of problem.score
    calculated_ratios = [] # Stores the correctly calculated approximation ratios

    model.eval() # Set model to evaluation mode
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            print(f"Processing Batch {batch_idx + 1}/{len(test_loader)}...")
            # The original script processes one example at a time from the batch
            # This is fine for ForcedRB where batch_size is often 1 for testing detailed examples
            for example_idx_in_batch, example in enumerate(batch.to_data_list()):
                if total_count > 0 and total_count % 50 == 0: # Print progress
                    print(f"  Processed {total_count} examples...")

                start_time = time.time()

                # For Vertex Cover, problem.score usually returns -cover_size.
                # So, a higher score (closer to 0) is better.
                # We initialize best_raw_score to negative infinity to find the "highest" score.
                best_raw_score = -math.inf

                # Perform 100 attempts as per paper's methodology for robust testing
                for attempt in range(100):
                    # It's important that featurize_batch handles individual examples correctly
                    # or that the model input is prepared appropriately for single examples.
                    # The original script passed 'example' directly.
                    x_in, current_example_state = featurize_batch(args, example.clone()) # Clone to avoid in-place modification issues

                    x_out = model(x_in, current_example_state) # Pass the featurized example state
                    # Loss calculation (optional for final scoring, but good for consistency)
                    # loss = problem.loss(x_out, current_example_state)
                    # total_loss += float(loss) # If you decide to track it

                    # Assuming random_hyperplane_projector works with single examples
                    x_proj = random_hyperplane_projector(args, x_out, current_example_state, problem.score)

                    # ENSURE we are getting a +/- 1 vector out by replacing 0 with 1
                    # For vertex cover, this might be 0 (not in VC) or 1 (in VC)
                    # If it's +/- 1, it needs to be mapped to 0/1 for VC size calculation.
                    # This part needs to align with how problem.score expects the input.
                    # Let's assume problem.score handles the +/-1 to 0/1 mapping if necessary.
                    x_proj = torch.where(x_proj == 0, torch.tensor(1.0, device=x_proj.device), x_proj) # Avoid 0 if it's problematic

                    num_zeros = (x_proj == 0).count_nonzero()
                    # For VC, 0 might be a valid state (not in cover). This assert might need adjustment
                    # depending on the output representation of x_proj and how problem.score interprets it.
                    # If x_proj is truly +/-1, then `assert num_zeros == 0` is correct.

                    # Get the raw score from the problem's score function
                    current_raw_score = problem.score(args, x_proj, current_example_state)
                    best_raw_score = max(current_raw_score, best_raw_score)

                end_time = time.time()
                times.append(end_time - start_time)

                scores_from_problem_score.append(float(best_raw_score))

                # --- Vertex Cover Specific Ratio Calculation ---
                if args.problem_type == 'vertex_cover':
                    # Assuming problem.score returns -cover_size for vertex cover
                    achieved_cover_size = -float(best_raw_score)
                    # `example.optimal` from forced_rb_dataset.py is alpha(G) (max independent set size)
                    optimal_independent_set_size = example.optimal
                    optimal_vertex_cover_size = example.num_nodes - optimal_independent_set_size

                    if optimal_vertex_cover_size > 0:
                        current_ratio = achieved_cover_size / optimal_vertex_cover_size
                    elif achieved_cover_size == 0: # Optimal and achieved are both 0 (e.g., graph with no edges)
                        current_ratio = 1.0
                    else: # Optimal is 0, but achieved is > 0
                        current_ratio = float('inf') # Or handle as a very bad ratio

                    calculated_ratios.append(current_ratio)
                    print(f"  Example {total_count + 1}: Nodes={example.num_nodes}, OptIndSet={optimal_independent_set_size}, OptVC={optimal_vertex_cover_size}, AchievedVC={achieved_cover_size:.2f}, Ratio={current_ratio:.4f}, Time={(end_time - start_time):.3f}s")
                else:
                    # Fallback for other problems (original ratio, may need adjustment for them too)
                    # This part is likely for Max-Cut where higher score is better and optimal is max_cut_value
                    if (example.num_nodes - example.optimal) != 0:
                         original_ratio_calc = -float(best_raw_score) / (example.num_nodes - example.optimal)
                    else: # Avoid division by zero
                        original_ratio_calc = float('nan') # Or some other indicator
                    calculated_ratios.append(original_ratio_calc)
                    print(f"  Example {total_count + 1} ({args.problem_type}): RawScore={best_raw_score:.2f}, OriginalRatioCalc={original_ratio_calc:.4f}, Time={(end_time - start_time):.3f}s")


                total_count += 1

                if stop_early and total_count >= 5: # For quick testing of the script
                    print("Stopping early due to stop_early=True and 5 examples processed.")
                    return scores_from_problem_score, calculated_ratios, times


    return scores_from_problem_score, calculated_ratios, times


if __name__ == '__main__':
    args = parse_test_args()
    print("Parsed Arguments:")
    for arg, value in sorted(vars(args).items()):
        print(f"  {arg}: {value}")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed) # Also seed numpy for data generation if it uses numpy.random
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)


    # get data, model
    # construct_loaders should use args.dataset, args.gen_n, args.gen_k etc.
    # (read from params.txt in model_folder) to generate ForcedRB instances.
    # It also needs to populate example.optimal correctly.
    # The paper mentions 1000 test graphs. Ensure construct_loaders provides this.
    # args.num_graphs from params.txt should guide this.
    print(f"Constructing test loader for dataset: {args.dataset} (mode=test)...")
    if args.dataset not in test_datasets: # This branch is usually for generated datasets
        if args.use_val_set: # Should not be true for final testing
            print("Warning: Using validation set for testing. This is unusual for final evaluation.")
            _, test_loader, _ = construct_loaders(args)
        else:
            # This should generate 1000 test graphs based on params.txt (args.num_graphs)
            # and gen_n, gen_k from params.txt
            _, _, test_loader = construct_loaders(args)
    else: # This branch is for pre-defined test datasets like TU-datasets
        test_loader = construct_loaders(args, mode="test")
    print(f"Test loader constructed. Estimated number of examples: {len(test_loader.dataset) if hasattr(test_loader, 'dataset') else 'N/A (IterableDataset)'}")

    print("Constructing model...")
    model, _ = construct_model(args) # Uses params from model_folder
    print("Getting problem definition...")
    problem = get_problem(args) # Uses problem_type from model_folder

    model_path = os.path.join(args.model_folder, args.model_file)
    print(f"Loading model from: {model_path}")
    model = load_model(model, model_path)
    model.to(args.device)
    print(f"Model loaded and moved to {args.device}.")

    print("Running initial CUDA synchronization pass (stop_early=True)...")
    # To initialize CUDA, etc. Run on a very small number of examples.
    # Modifying stop_early to ensure it actually stops early for this pass.
    _ = time_and_scores(args, model, test_loader, problem, stop_early=True)
    print("CUDA sync pass complete. Starting full evaluation...")

    predictions = time_and_scores(args, model, test_loader, problem, stop_early=args.stop_early_testing) # Use a new arg for full run stop_early
    raw_scores, final_ratios, execution_times = predictions

    if final_ratios:
        print(f"\n--- Final Results ({args.problem_type}) ---")
        print(f"Number of test instances evaluated: {len(final_ratios)}")
        print(f"Mean Raw Score (from problem.score): {sum(raw_scores) / len(raw_scores):.4f}")
        print(f"Variance Raw Score: {np.var(raw_scores):.4f}")
        print(f"Mean Approximation Ratio: {sum(final_ratios) / len(final_ratios):.4f}")
        print(f"Std Dev of Approximation Ratio: {np.std(final_ratios):.4f}") # Standard deviation is more common than variance for ratios
        print(f"Total testing time (excluding CUDA sync): {sum(execution_times):.2f}s")
        print(f"Average time per instance: {(sum(execution_times) / len(execution_times) if execution_times else 0):.3f}s")
    else:
        print("No results to report. Check if test_loader was empty or stop_early prevented full run.")


    # Save the raw scores and the calculated ratios
    output_data_to_save = {
        'raw_scores': np.array(raw_scores),
        'calculated_ratios': np.array(final_ratios),
        'execution_times': np.array(execution_times)
    }
    output_filename = os.path.join(args.model_folder, f'{args.test_prefix}@@VC_test_results_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}.npz')
    np.savez(output_filename, **output_data_to_save)
    print(f"Results saved to: {output_filename}")

    print("Finished predicting!")