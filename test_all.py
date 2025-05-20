import os
import subprocess
import json
import pandas as pd
from datetime import datetime
import traceback # For detailed error tracebacks

# --- Configuration ---
BASE_TRAINING_RUNS_DIR = "training_runs" 
TEST_SCRIPT_PATH = "test_general.py"    # Path to your generic testing script
COMMON_MODEL_FILENAME = "best_model.pt" # Common name for the model file to test
PARAMS_FILENAME = "params.txt"

# Optional: Specific subdirectories within BASE_TRAINING_RUNS_DIR to scan
TARGET_SUBDIRS = None 
# Example: TARGET_SUBDIRS = ["vertex_cover_experiments", "max_cut_models"]


# CLI arguments to pass to each invocation of test_general.py
# If empty, test_general.py will use its own defaults (e.g., hardcoded MAX_BATCHES_FOR_QUICK_TEST if applicable)
# Example: To make test_general.py run its --stop-early-testing logic (e.g. 5 examples):
# CLI_ARGS_FOR_TEST_SCRIPT = ["--stop-early-testing"] 
# Example: If test_general.py supports --max-batches for more control:
# CLI_ARGS_FOR_TEST_SCRIPT = ["--max-batches", "3"]
CLI_ARGS_FOR_TEST_SCRIPT = [] # Current default: relies on test_general.py's internal logic

ERROR_LOG_DIR = "test_all_errors" # Directory to store error logs
os.makedirs(ERROR_LOG_DIR, exist_ok=True) 
ERROR_LOG_FILE = os.path.join(ERROR_LOG_DIR, f"run_all_tests_errors_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

# --- Helper Functions ---

def find_model_folders(base_dir, target_subdirs=None):
    """Finds all valid model folders to test."""
    model_folders = []
    scan_paths = [os.path.join(base_dir, subdir) for subdir in target_subdirs] if target_subdirs else [base_dir]

    for scan_path in scan_paths:
        if not os.path.isdir(scan_path):
            print(f"Warning: Scan path {scan_path} is not a directory. Skipping.")
            continue
        # We walk topdown to modify dirs list and prevent descending into already found model folders
        for root, dirs, files in os.walk(scan_path, topdown=True): 
            if PARAMS_FILENAME in files and COMMON_MODEL_FILENAME in files:
                # This 'root' is a potential model folder
                model_folders.append(root)
                # Don't descend further into subdirectories of this found model folder
                dirs[:] = [] 
            # else:
                # If you have a structure like base_dir/problem_type/dataset_name/paramhash
                # you might need more complex logic to identify the correct "leaf" model folders.
                # The current logic assumes params.txt and model.pt are directly in the folder to test.
    return model_folders

def log_error(model_folder_path, command_str_list, stdout, stderr, exception_info=""):
    """Logs error details to the error log file."""
    command_str = ' '.join(command_str_list) # Convert list to string for logging
    with open(ERROR_LOG_FILE, 'a') as f:
        f.write(f"--- ERROR: Testing Model: {model_folder_path} ---\n")
        f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Command: {command_str}\n")
        if stdout:
            f.write("Stdout:\n")
            f.write(stdout + "\n")
        if stderr:
            f.write("Stderr:\n")
            f.write(stderr + "\n")
        if exception_info:
            f.write("Python Exception Traceback / Info:\n")
            f.write(exception_info + "\n")
        f.write("-" * 70 + "\n\n")

def run_single_test(model_folder_path, test_script, model_filename, cli_args_for_test_script):
    """
    Runs the test script for a single model and extracts key results.
    Returns a dictionary of results or None if the test fails or results can't be parsed.
    """
    print(f"\n--- Testing model in: {model_folder_path} ---")
    
    command = [
        "python", test_script,
        "--model-folder", model_folder_path,
        "--model-file", model_filename
    ]
    command.extend(cli_args_for_test_script)
    process_info = None 
    run_stdout, run_stderr = "", ""


    try:
        # Timeout in seconds (e.g., 30 minutes = 1800 seconds)
        # Adjust as needed, or make it configurable
        timeout_seconds = 1800 
        process = subprocess.run(command, capture_output=True, text=True, check=False, timeout=timeout_seconds)
        process_info = process 
        run_stdout = process.stdout
        run_stderr = process.stderr

        if process.returncode != 0:
            print(f"ERROR: Test script failed for {model_folder_path} with return code {process.returncode}")
            log_error(model_folder_path, command, run_stdout, run_stderr)
            return None

        print(f"Successfully ran test for {model_folder_path}")
        
        # Find the latest JSON results file
        result_files = []
        for f_name in os.listdir(model_folder_path):
            if f_name.endswith('.json') and '_results__' in f_name:
                result_files.append(os.path.join(model_folder_path, f_name))
        
        if not result_files:
            print(f"ERROR: No JSON result file found in {model_folder_path} after test.")
            log_error(model_folder_path, command, run_stdout, run_stderr, "No JSON result file found.")
            return None
        
        latest_result_file = max(result_files, key=os.path.getctime)
        print(f"Parsing results from: {latest_result_file}")

        with open(latest_result_file, 'r') as f:
            results_data = json.load(f)
        
        args_from_run = results_data.get('args', {})
        summary_stats = results_data.get('summary_stats', {})

        problem_type = args_from_run.get('problem_type', 'UnknownProblem')
        dataset_name = args_from_run.get('dataset', 'UnknownDataset')
        model_type = args_from_run.get('model_type', 'UnknownModel') # from params.txt via args
        
        mean_ratio = summary_stats.get('mean_ratio (valid)')
        avg_time_per_instance = summary_stats.get('avg_time_per_instance_sec')
        num_instances_evaluated = summary_stats.get('num_instances_evaluated', 0)
        mean_raw_score = summary_stats.get('mean_raw_score')

        return {
            "model_folder_leaf": os.path.basename(model_folder_path), 
            "problem_type": problem_type,
            "model_type": model_type,
            "dataset_name": dataset_name,
            "num_instances_evaluated": num_instances_evaluated,
            "mean_approximation_ratio": mean_ratio if mean_ratio is not None else 'N/A',
            "mean_raw_score": mean_raw_score if mean_raw_score is not None else 'N/A',
            "avg_time_per_instance_sec": avg_time_per_instance if avg_time_per_instance is not None else 'N/A',
            "detailed_results_file": os.path.basename(latest_result_file)
        }

    except subprocess.TimeoutExpired:
        print(f"ERROR: Test script TIMED OUT for {model_folder_path} (limit: {timeout_seconds}s)")
        log_error(model_folder_path, command, run_stdout, run_stderr, f"Subprocess TimeoutExpired after {timeout_seconds}s")
        return None
    except FileNotFoundError:
        error_msg = f"Test script '{test_script}' not found. Cannot proceed with batch testing."
        print(f"CRITICAL ERROR: {error_msg}")
        log_error("GLOBAL_SETUP_ERROR", command, "", "", error_msg)
        return "STOP_BATCH" # Special return to stop the whole batch
    except Exception as e:
        print(f"ERROR: An unhandled exception occurred while processing {model_folder_path}: {e}")
        # Use stdout/stderr from process_info if available, otherwise indicate they might not be
        stdout_to_log = process_info.stdout if process_info and hasattr(process_info, 'stdout') else run_stdout
        stderr_to_log = process_info.stderr if process_info and hasattr(process_info, 'stderr') else run_stderr
        if not stdout_to_log and not stderr_to_log:
            stdout_to_log = "N/A (Exception likely before subprocess or during parsing)"
        log_error(model_folder_path, command, stdout_to_log, stderr_to_log, traceback.format_exc())
        return None

# --- Main Orchestration ---
def main():
    print(f"Starting batch testing of models. Error log will be at: {ERROR_LOG_FILE}")
    model_folders_to_test = find_model_folders(BASE_TRAINING_RUNS_DIR, TARGET_SUBDIRS)
    
    if not model_folders_to_test:
        print(f"No model folders found matching criteria in '{BASE_TRAINING_RUNS_DIR}' (and subdirs '{TARGET_SUBDIRS if TARGET_SUBDIRS else 'None'}'). Exiting.")
        return

    print(f"Found {len(model_folders_to_test)} model folder(s) to test.")
    all_results_summary = []
    successful_tests = 0
    failed_tests = 0

    for i, folder_path in enumerate(model_folders_to_test):
        # Construct a more descriptive name for printing, relative to BASE_TRAINING_RUNS_DIR
        display_folder_path = os.path.relpath(folder_path, start=os.path.commonpath([BASE_TRAINING_RUNS_DIR, folder_path]))
        if not display_folder_path.startswith(os.path.basename(BASE_TRAINING_RUNS_DIR)): # if commonpath was just '.'
            display_folder_path = os.path.join(os.path.basename(BASE_TRAINING_RUNS_DIR), display_folder_path)

        print(f"\nProcessing folder {i+1}/{len(model_folders_to_test)}: {display_folder_path}")
        
        summary = run_single_test(folder_path, TEST_SCRIPT_PATH, COMMON_MODEL_FILENAME, CLI_ARGS_FOR_TEST_SCRIPT)
        
        if summary == "STOP_BATCH":
            print("Stopping batch process due to critical setup error (e.g., test script not found).")
            break 
        elif summary:
            all_results_summary.append(summary)
            successful_tests += 1
            # Provide a more informative success message
            ratio = summary['mean_approximation_ratio']
            time_inst = summary['avg_time_per_instance_sec']
            print(f"  SUCCESS: Problem='{summary['problem_type']}', Dataset='{summary['dataset_name']}', Model='{summary['model_type']}', Ratio={ratio}, Time/Inst={time_inst}s")
        else:
            failed_tests += 1
            print(f"  FAILURE: Check '{ERROR_LOG_FILE}' for details on model in '{display_folder_path}'.")

    print("\n--- Batch Testing Summary ---")
    print(f"Total models attempted: {len(model_folders_to_test)}") # This is the number found
    print(f"Successful tests: {successful_tests}")
    print(f"Failed tests (or stopped): {failed_tests + (len(model_folders_to_test) - successful_tests - failed_tests if summary == 'STOP_BATCH' else 0 ) }")


    if not all_results_summary:
        print("No results were successfully collected from any test run.")
        return

    summary_df = pd.DataFrame(all_results_summary)
    if not summary_df.empty:
        summary_df = summary_df.sort_values(by=['problem_type', 'dataset_name', 'model_type'])
    
    # Create a more structured filename for the summary
    timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    config_parts = []
    if TARGET_SUBDIRS:
        config_parts.append("subdirs_" + "_".join(TARGET_SUBDIRS))
    if CLI_ARGS_FOR_TEST_SCRIPT:
        sanitized_cli = "_".join(arg.replace("--", "").replace("=", "_").replace(" ", "") for arg in CLI_ARGS_FOR_TEST_SCRIPT)
        config_parts.append(f"cli_{sanitized_cli}")
    
    config_str = "_".join(config_parts) if config_parts else "defaultconfig"
    summary_filename_base = f"all_models_summary_{config_str}_{timestamp_str}"
    
    summary_csv_path = f"{summary_filename_base}.csv"
    summary_json_path = f"{summary_filename_base}.json" # For JSON fallback

    try:
        if not summary_df.empty:
            summary_df.to_csv(summary_csv_path, index=False)
            print(f"\nAggregated summary saved to: {summary_csv_path}")
        else:
            print("\nNo successful tests to save in summary CSV.")
    except Exception as e:
        print(f"Error saving summary CSV to '{summary_csv_path}': {e}")
        if not summary_df.empty:
            print("Printing summary to console instead:")
            print(summary_df.to_string())
        # Save as JSON as a fallback
        try:
            with open(summary_json_path, 'w') as f:
                json.dump(all_results_summary, f, indent=4) # Save the list of dicts
            print(f"Fallback summary JSON saved to: {summary_json_path}")
        except Exception as json_e:
            print(f"Error saving fallback summary JSON to '{summary_json_path}': {json_e}")


    print("\nBatch testing complete.")

if __name__ == "__main__":
    main()