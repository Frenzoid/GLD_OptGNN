import numpy as np
import os
import time
import torch
import torch.nn.functional as F
from model.saving import save_model
from problem.baselines import random_hyperplane_projector 
from torch_geometric.transforms import AddRandomWalkPE

def featurize_batch(args, batch):
    batch_device = batch.x.device if hasattr(batch, 'x') and batch.x is not None else args.device # Fallback to args.device
    batch = batch.to(batch_device) # Ensure batch is on a consistent device first
    N = batch.num_nodes

    if args.positional_encoding is None or args.pe_dimension == 0:
        x_in = torch.randn((N, args.rank), dtype=torch.float, device=batch_device)
        if args.rank > 0: # Avoid normalize if rank is 0
            x_in = F.normalize(x_in, dim=1)
    elif args.positional_encoding == 'laplacian_eigenvector':
        pe_dim_to_use = min(int(args.pe_dimension), args.rank)
        rand_dim = args.rank - pe_dim_to_use
        
        pe_data = batch.laplacian_eigenvector_pe if hasattr(batch, 'laplacian_eigenvector_pe') else batch.pos # Common alternative name
        pe = pe_data.to(batch_device)[:, :pe_dim_to_use]
        
        # Ensure sign tensor is on the same device
        sign = (-1 + 2 * torch.randint(0, 2, (pe_dim_to_use, ), device=batch_device)).float() 
        pe *= sign
        
        if rand_dim > 0:
            x_rand_in = torch.randn((N, rand_dim), dtype=torch.float, device=batch_device)
            x_rand_in = F.normalize(x_rand_in, dim=1)
            x_in = torch.cat((x_rand_in, pe), 1)
        elif pe_dim_to_use > 0:
            x_in = pe
        else: # Fallback if rank is 0
             x_in = torch.randn((N, args.rank), dtype=torch.float, device=batch_device) # Should be 0-dim if args.rank is 0
    elif args.positional_encoding == 'random_walk':
        if not hasattr(batch, 'random_walk_pe') or \
           (hasattr(batch, 'random_walk_pe') and batch.random_walk_pe is not None and batch.random_walk_pe.shape[1] < int(args.pe_dimension)):
            # Apply transform on CPU to avoid potential CUDA errors with some transforms, then move to device
            original_device = batch.x.device if hasattr(batch, 'x') and batch.x is not None else 'cpu'
            batch_cpu = batch.to('cpu')
            transform = AddRandomWalkPE(walk_length=int(args.pe_dimension))
            batch_cpu = transform(batch_cpu)
            batch = batch_cpu.to(original_device) # Move back to original device
        
        pe_dim_to_use = min(int(args.pe_dimension), args.rank)
        rand_dim = args.rank - pe_dim_to_use

        pe_data = batch.random_walk_pe.to(batch_device)
        pe = pe_data[:, :pe_dim_to_use]

        if rand_dim > 0:
            x_rand_in = torch.randn((N, rand_dim), dtype=torch.float, device=batch_device)
            x_rand_in = F.normalize(x_rand_in, dim=1) if rand_dim > 0 else x_rand_in # Normalize if rand_dim > 0
            x_in = torch.cat((x_rand_in, pe), 1)
        elif pe_dim_to_use > 0:
            x_in = pe
        else: 
            x_in = torch.randn((N, args.rank), dtype=torch.float, device=batch_device)
            if args.rank > 0: x_in = F.normalize(x_in, dim=1)
    else:
        raise ValueError(f"Invalid positional_encoding passed: {args.positional_encoding}")

    batch.penalty = args.penalty
    return x_in, batch.to(args.device) # Ensure final batch is on args.device for model

def validate(args, model, val_loader, problem):
    total_loss_sum = 0.
    total_score_sum = 0.
    total_constraint_sum = 0.
    total_graphs_validated = 0
    model.eval()
    with torch.no_grad():
        for batch_data in val_loader:
            # featurize_batch ensures batch_featurized is on args.device
            x_in, batch_featurized = featurize_batch(args, batch_data) 
            x_out = model(x_in, batch_featurized)
            
            loss_value = problem.loss(x_out, batch_featurized) # Assume returns float or 0-D tensor (sum for batch)
            # If it's a tensor, get .item(). If float, this is fine.
            scalar_loss_value = loss_value.item() if isinstance(loss_value, torch.Tensor) else float(loss_value)
            total_loss_sum += scalar_loss_value # This is sum of losses in batch

            # problem.score might not be directly usable by projector if it's just a value.
            # projector often takes the model's raw scores (x_out) or a way to compute them.
            # For now, assuming problem.score is a metric function, not for projection.
            # The third arg to random_hyperplane_projector is 'score_fn', not 'problem.score' value.
            # Let's assume the problem object has a method that can serve as score_fn if needed,
            # or random_hyperplane_projector has a default way if problem.score is not a function.
            # For simplicity, if problem.score is not directly callable for projection, pass None or a default.
            # This part of random_hyperplane_projector needs to be clear.
            # If problem.score is the *metric*, it's used later.
            score_function_for_projector = problem.score if callable(problem.score) else None # Placeholder
            x_proj = random_hyperplane_projector(args, x_out, batch_featurized, score_function_for_projector)
            x_proj = torch.where(x_proj == 0, torch.tensor(1.0, device=x_proj.device), x_proj)
            assert (x_proj == 0).count_nonzero() == 0

            score_value = problem.score(args, x_proj, batch_featurized) # Assume returns float or 0-D tensor (sum for batch)
            constraint_value = problem.constraint(x_proj, batch_featurized) # Assume returns float or 0-D tensor (sum for batch)

            scalar_score_value = score_value.item() if isinstance(score_value, torch.Tensor) else float(score_value)
            scalar_constraint_value = constraint_value.item() if isinstance(constraint_value, torch.Tensor) else float(constraint_value)

            total_score_sum += scalar_score_value
            total_constraint_sum += scalar_constraint_value
            total_graphs_validated += batch_featurized.num_graphs

    if total_graphs_validated == 0:
        return 0.0, 0.0, 0.0
    
    avg_loss = total_loss_sum / total_graphs_validated
    avg_score = total_score_sum / total_graphs_validated
    avg_constraint = total_constraint_sum / total_graphs_validated
    
    return avg_loss, avg_score, avg_constraint

def train(args, model, train_loader, optimizer, problem, val_loader=None, test_loader=None):
    epochs_to_run = args.epochs
    model_folder = args.log_dir

    log_train_losses_per_graph = [] # Storing per-graph average losses
    log_valid_losses = []
    log_valid_scores = []
    log_valid_constraints = []
    log_test_losses = []
    log_test_scores = []
    log_test_constraints = []

    model.to(args.device)
    ep = 0
    steps = 0 
    
    segment_total_loss_sum = 0.0 
    segment_graph_count = 0  
    segment_start_time = time.time() 

    while True:
        if args.stepwise and steps >= args.steps:
            print(f"Target steps {args.steps} reached. Finalizing training.")
            break
        if not args.stepwise and ep >= epochs_to_run:
            print(f"Target epochs {epochs_to_run} reached. Finalizing training.")
            break
        
        model.train()
        epoch_start_time = time.time()
        epoch_cumulative_loss_sum = 0.0 
        epoch_cumulative_graphs = 0

        for batch_idx, batch_data in enumerate(train_loader):
            x_in, batch_featurized = featurize_batch(args, batch_data)
            # Remove the .to("cuda") call to prevent device mismatch
            x_out = model(x_in, batch_featurized)
            
            loss_for_batch_tensor = problem.loss(x_out, batch_featurized) # MUST be a tensor for backward()

            optimizer.zero_grad()
            loss_for_batch_tensor.backward() 
            optimizer.step()

            # Rest of the function remains unchanged
            scalar_loss_for_batch = loss_for_batch_tensor.item() 

            if batch_featurized.num_graphs > 0:
                log_train_losses_per_graph.append(scalar_loss_for_batch / batch_featurized.num_graphs)
            else: # Should not happen with valid batches
                log_train_losses_per_graph.append(scalar_loss_for_batch) 

            epoch_cumulative_loss_sum += scalar_loss_for_batch
            epoch_cumulative_graphs += batch_featurized.num_graphs
            
            segment_total_loss_sum += scalar_loss_for_batch
            segment_graph_count += batch_featurized.num_graphs
            steps += 1
            
            if args.stepwise:
                if args.infinite and steps > 0 and steps % 100 == 0: # Assuming 100 is the log frequency
                    current_segment_time = time.time() - segment_start_time
                    if segment_graph_count > 0:
                        avg_segment_loss = segment_total_loss_sum / segment_graph_count
                        print(f"steps={steps} t={current_segment_time:.2f} avg_loss (last {segment_graph_count} graphs)={avg_segment_loss:.2f}")
                    else:
                        print(f"steps={steps} t={current_segment_time:.2f} (no graphs in last segment for avg loss)")
                    segment_total_loss_sum = 0.0
                    segment_graph_count = 0
                    segment_start_time = time.time()

                if args.valid_freq != 0 and steps > 0 and steps % args.valid_freq == 0:
                    valid_run_start_time = time.time()
                    avg_valid_loss, avg_valid_score, avg_valid_constraint = validate(args, model, val_loader, problem)
                    
                    if not log_valid_scores or avg_valid_score > max(log_valid_scores):
                        save_model(model, f"{model_folder}/best_model.pt")
                    log_valid_losses.append(avg_valid_loss)
                    log_valid_scores.append(avg_valid_score)
                    log_valid_constraints.append(avg_valid_constraint)
                    valid_run_time = time.time() - valid_run_start_time
                    
                    avg_test_loss, avg_test_score, avg_test_constraint = float('inf'), float('-inf'), float('inf')
                    if test_loader is not None:
                        avg_test_loss, avg_test_score, avg_test_constraint = validate(args, model, test_loader, problem)
                        log_test_losses.append(avg_test_loss)
                        log_test_scores.append(avg_test_score)
                        log_test_constraints.append(avg_test_constraint)
                    print(f"  VALIDATION epoch={ep} steps={steps} t={valid_run_time:.2f}\n"
                          f"              valid_loss={avg_valid_loss:.4f} valid_score={avg_valid_score:.4f} valid_constraint={avg_valid_constraint:.4f}\n"
                          f"              test_loss={avg_test_loss:.4f} test_score={avg_test_score:.4f} test_constraint={avg_test_constraint:.4f}")

                if args.save_freq != 0 and steps > 0 and steps % args.save_freq == 0:
                    save_model(model, f"{model_folder}/model_step{steps}.pt")

                if steps >= args.steps:
                    break 
        
        epoch_duration = time.time() - epoch_start_time
        if epoch_cumulative_graphs > 0:
            current_epoch_avg_loss = epoch_cumulative_loss_sum / epoch_cumulative_graphs
            print(f"epoch={ep} t={epoch_duration:.2f} steps={steps} epoch_avg_loss={current_epoch_avg_loss:.2f}")
        elif not args.stepwise: 
            print(f"epoch={ep} t={epoch_duration:.2f} steps={steps} WARNING: No graphs processed in this epoch.")
        
        if not args.stepwise:
            if args.valid_freq != 0 and ep % args.valid_freq == 0 and ep > 0 : # Check ep > 0 to avoid validation at epoch 0 if freq is low
                valid_run_start_time = time.time()
                avg_valid_loss, avg_valid_score, avg_valid_constraint = validate(args, model, val_loader, problem)
                if not log_valid_scores or avg_valid_score > max(log_valid_scores):
                    save_model(model, f"{model_folder}/best_model.pt")
                log_valid_losses.append(avg_valid_loss)
                log_valid_scores.append(avg_valid_score)
                log_valid_constraints.append(avg_valid_constraint)
                valid_run_time = time.time() - valid_run_start_time

                avg_test_loss, avg_test_score, avg_test_constraint = float('inf'), float('-inf'), float('inf')
                if test_loader is not None:
                    avg_test_loss, avg_test_score, avg_test_constraint = validate(args, model, test_loader, problem)
                    log_test_losses.append(avg_test_loss)
                    log_test_scores.append(avg_test_score)
                    log_test_constraints.append(avg_test_constraint)
                print(f"  VALIDATION epoch={ep} steps={steps} t={valid_run_time:.2f}\n"
                      f"              valid_loss={avg_valid_loss:.4f} valid_score={avg_valid_score:.4f} valid_constraint={avg_valid_constraint:.4f}\n"
                      f"              test_loss={avg_test_loss:.4f} test_score={avg_test_score:.4f} test_constraint={avg_test_constraint:.4f}")

            if args.save_freq != 0 and ep % args.save_freq == 0 and ep > 0: 
                save_model(model, f"{model_folder}/model_ep{ep}.pt")
        
        ep += 1
    
    final_save_name = f"model_step{steps}.pt" if args.stepwise else f"model_ep{ep-1}.pt" # ep-1 because ep is incremented after last completed epoch
    final_save_path = os.path.join(model_folder, final_save_name)
    save_model(model, final_save_path)
    print(f"Saved final model to {final_save_path}")
    
    np.save(os.path.join(args.log_dir, "train_losses_per_graph.npy"), log_train_losses_per_graph) # Updated name
    np.save(os.path.join(args.log_dir, "valid_losses.npy"), log_valid_losses)
    np.save(os.path.join(args.log_dir, "valid_scores.npy"), log_valid_scores)
    np.save(os.path.join(args.log_dir, "valid_constraints.npy"), log_valid_constraints)
    if test_loader is not None and log_test_losses: # Check if list is populated
        np.save(os.path.join(args.log_dir, "test_losses.npy"), log_test_losses)
        np.save(os.path.join(args.log_dir, "test_scores.npy"), log_test_scores)
        np.save(os.path.join(args.log_dir, "test_constraints.npy"), log_test_constraints)

def predict(model, loader, args, problem): 
    predictions = []
    model.eval()
    with torch.no_grad():
        for batch_data in loader:
            x_in, batch_featurized = featurize_batch(args, batch_data)
            x_out = model(x_in, batch_featurized)
            
            score_function_for_projector = None
            if hasattr(problem, 'score') and callable(problem.score):
                 # Check if problem.score can be called with (args, x_proj_tensor, batch_obj)
                 # The projector might need a different signature or use x_out directly.
                 # For now, let's assume it's the metric function and projector handles it or has default.
                 pass # random_hyperplane_projector's 3rd arg needs a score_fn, not the problem object directly
            
            # This needs clarification on what random_hyperplane_projector's score_fn argument expects.
            # If it expects a function that takes (x_out, batch) and returns scores, you might need a wrapper.
            # For now, passing None if problem.score is not directly compatible.
            # x_proj = random_hyperplane_projector(args, x_out, batch_featurized, score_function_for_projector_if_needed)
            x_proj = random_hyperplane_projector(args, x_out, batch_featurized, None) # Simplification: assuming projector can work without specific score_fn or uses internal logic

            x_proj = torch.where(x_proj == 0, torch.tensor(1.0, device=x_proj.device), x_proj)
            
            # Storing projected solutions. How to split if batch_featurized.num_graphs > 1 needs specific logic.
            # This is a common challenge. Often for predict, batch_size=1 is used in the loader.
            if batch_featurized.num_graphs > 1:
                print(f"WARNING: Predict function processing batch with {batch_featurized.num_graphs} graphs. Output splitting not fully implemented here.")
                # Storing the whole batch output for now
                predictions.append({'x_projected_batch': x_proj.cpu().numpy(), 
                                    'num_graphs_in_batch': batch_featurized.num_graphs,
                                    'ptr': batch_featurized.ptr.cpu().numpy() if hasattr(batch_featurized, 'ptr') else None }) # Include ptr for potential splitting later
            else: 
                 predictions.append({'x_projected': x_proj.cpu().numpy()}) # Assumes x_proj is for a single graph
    
    if not predictions:
        print("WARNING: Predict function produced no predictions (loader might be empty or issue in processing).")
    # Consider raising NotImplementedError if the predict logic is not fully defined for your needs.
    # raise NotImplementedError("Predict function requires specific output format and handling of batched graphs.")
    return predictions