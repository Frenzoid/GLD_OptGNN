import os
import json
import pandas as pd
import argparse
from pathlib import Path

def determine_experiment_type(folder_name):
    """Determine if an experiment is normal, TU, or Ablation based on the naming pattern."""
    # The list of TU dataset names
    TU_DATASETS = ['COLLAB', 'ENZYMES', 'IMDB-BINARY', 'MUTAG', 'PROTEINS']
    
    # List of known problem prefixes
    PROBLEM_PREFIXES = ['max_cut', 'sat_random', 'vertex_cover', 'max_sat']
    
    # Find which prefix this folder has (if any)
    prefix = None
    for p in PROBLEM_PREFIXES:
        if folder_name.startswith(p):
            prefix = p
            break
    
    # If no known prefix found, try to determine based on first part before underscore
    if prefix is None and '_' in folder_name:
        prefix = folder_name.split('_')[0]
    
    # For folders like "sat_random-sat_r32", we need special handling
    if prefix == 'sat':
        # For sat problems, the ablation naming pattern is different
        if '-' in folder_name and '_' in folder_name:
            return 'Normal'  # sat_random-sat_r32 format
    
    # Check if it's an ablation (no underscore after the prefix)
    if prefix and prefix in folder_name:
        remainder = folder_name[len(prefix):]
        # If the remainder doesn't start with underscore but has letters right after the prefix
        if remainder and not remainder.startswith('_') and any(c.isalpha() for c in remainder[:3]):
            return 'Ablation'  # Like max_cutGCNNBA100200
    
    # Check if it's a TU dataset
    for tu_name in TU_DATASETS:
        if tu_name in folder_name:
            return 'TU'
    
    # Otherwise, it's a normal experiment
    return 'Normal'
    
def process_json_file(file_path):
    """Extract args and summary_stats from a JSON file."""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Extract required sections
        args = data.get('args', {})
        summary_stats = data.get('summary_stats', {})
        
        # Get folder name from the parent directory (ignoring the param hash directory)
        folder_name = file_path.parent.parent.name
        
        # Determine experiment type
        experiment_type = determine_experiment_type(folder_name)
        
        # Create a flat dictionary combining both sections
        result = {
            'file_path': str(file_path),
            'folder_name': folder_name,
            'param_hash': file_path.parent.name,
            'experiment_type': experiment_type
        }
        
        # Add args with prefix
        for key, value in args.items():
            result[f'args_{key}'] = value
            
        # Add summary stats with prefix
        for key, value in summary_stats.items():
            result[f'stats_{key}'] = value
            
        return result
    
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def find_json_files(directory):
    """Recursively find all JSON files in a directory."""
    json_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.json'):
                json_files.append(Path(root) / file)
    return json_files

def main():
    parser = argparse.ArgumentParser(description='Process OptGNN result JSON files.')
    parser.add_argument('--dir', type=str, required=True, help='Root directory to scan for JSON files')
    parser.add_argument('--output', type=str, default='results_summary.csv', help='Output CSV file')
    parser.add_argument('--filter', type=str, choices=['all', 'Normal', 'TU', 'Ablation'], default='all', 
                        help='Filter by experiment type')
    args = parser.parse_args()
    
    # Find all JSON files
    print(f"Scanning directory: {args.dir}")
    json_files = find_json_files(args.dir)
    print(f"Found {len(json_files)} JSON files")
    
    # Process each file
    results = []
    for file_path in json_files:
        result = process_json_file(file_path)
        if result:
            results.append(result)
    
    if not results:
        print("No valid results found!")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Filter by experiment type if requested
    if args.filter != 'all':
        df = df[df['experiment_type'] == args.filter]
        print(f"Filtered to {len(df)} {args.filter} experiments")
    
    # Save to CSV
    df.to_csv(args.output, index=False)
    print(f"Results saved to {args.output}")
    
    # Also save separate files by experiment type
    for exp_type in df['experiment_type'].unique():
        type_df = df[df['experiment_type'] == exp_type]
        type_output = args.output.replace('.csv', f'_{exp_type.lower()}.csv')
        type_df.to_csv(type_output, index=False)
        print(f"{exp_type} results ({len(type_df)} entries) saved to {type_output}")
    
    # Print summary
    print(f"\nProcessed {len(results)} JSON files successfully")
    print(f"Experiment type breakdown:")
    for exp_type, count in df['experiment_type'].value_counts().items():
        print(f"- {exp_type}: {count} experiments")
    
    # Print some useful stats
    if 'stats_mean_raw_score' in df.columns:
        print("\nMean score by experiment type:")
        for exp_type, group in df.groupby('experiment_type'):
            if 'stats_mean_raw_score' in group.columns:
                mean_score = group['stats_mean_raw_score'].mean()
                print(f"- {exp_type}: {mean_score:.2f}")

if __name__ == "__main__":
    main()