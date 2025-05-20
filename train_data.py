from itertools import product
from train import main  # assumes main(overrides: dict) is defined in train.py
import concurrent.futures as cf

# Configuration values
models = ['LiftMP']
datasets = ['ErdosRenyi','BarabasiAlbert','WattsStrogatz', 'PowerlawCluster']
gen_n_values = [(50, 100), (100, 200), (400, 500)]
r_values = [16]
lift_layers_values = [8]
pe_types = ['random_walk']
problems = ['max_cut','vertex_cover']

dataset_short_names = {
    'ErdosRenyi': 'ER',
    'BarabasiAlbert': 'BA',
    'WattsStrogatz': 'WS',
    'PowerlawCluster': 'HK',
}

for model, gen_n, r, lift_layers, pe, problem in product(
    models, gen_n_values, r_values, lift_layers_values, pe_types,problems):

    # with cf.ProcessPoolExecutor() as executor:
    # tasks = []
    for dataset in datasets:
        ds_short = dataset_short_names[dataset]
        gen_n_slug = str(gen_n[0])+'_'+str(gen_n[1])

        pe_dim = 8 if r == 32 else int(int(r) / 2)

        prefix = f"_{ds_short}_{gen_n_slug}"
        prefix =  problem + prefix 
        
        print(f"Running: {model} {dataset} R={r} Layers={lift_layers} GEN_N={gen_n_slug} PE={pe} Prefix={prefix}")

        overrides = {
            'stepwise': True,
            'steps': 20000,
            'valid_freq': 1000,
            'dropout': 0,
            'prefix': prefix,
            'model_type': model,
            'dataset': dataset,
            'parallel': 20,
            'infinite': True,
            'gen_n': list(gen_n),  # important: must be a list for multiple args
            'num_layers': lift_layers,
            'rank': r,
            'problem_type': problem,
            'batch_size': 16,
            'positional_encoding': pe,
            'pe_dimension': pe_dim,
            'lr': 0.001,
            # 'log_dir': f"logs/{prefix}",
            'seed': 42,
            # 'device': 'cuda',  # or 'cpu'
        }
        try:
            main(overrides)
            print("finished...")
        except Exception as e:
            print(f"Error occurred with item : {e}")

        # tasks.append(executor.submit(main, overrides))

        # for future in cf.as_completed(tasks):
        #     try:
        #         result = future.result()
        #         print("Task completed with result:", result)
        #     except Exception as e:
        #         print("Task failed with exception:", e)
        
# python test.py --model_folder="training_runs/max_cut_WS_50_100/paramhash:18cd86620d9b955ca2bf4942ef75f3f8a299a78116976f1e0bc11e4bdb8630a3" --model_file=best_model.pt --problem_type=max_cut --infinite=True --gen_n 50 100 --test_prefix=max_cut_WS_50_100