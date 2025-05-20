#!/usr/bin/env bash

set -e  # Stop on error

# Activate your local environment (adjust if needed)
# source ~/anaconda3/etc/profile.d/conda.sh
# conda activate your_env_name

for MODEL in 'LiftMP'; do
    for DATASET in 'arabasiAlbert' 'WattsStrogatz' 'HolmeKim'; do #for DATASET in 'ErdosRenyi' 'BarabasiAlbert' 'WattsStrogatz' 'HolmeKim'; do
        for GEN_N in "50 100" "100 200" "400 500"; do # ,  i skipped 50 100 for now 
            for R in  '16' ; do #for R in '4' '8' '16' '32'; do 
                for LIFT_LAYERS in '8' ; do #for LIFT_LAYERS in '8' '16'; do
                    for PE in 'random_walk' ; do

                        # Extract short dataset name
                        case $DATASET in
                            'ErdosRenyi') DS_SHORT="ER" ;;
                            'BarabasiAlbert') DS_SHORT="BA" ;;
                            'WattsStrogatz') DS_SHORT="WS" ;;
                            'HolmeKim') DS_SHORT="HK" ;;
                        esac

                        # Determine PE dimension
                        if [ "$R" == "32" ]; then
                            PE_DIM=8
                        else
                            PE_DIM=$((R / 2))
                        fi

                        PREFIX="max_cut_${DS_SHORT}_${GEN_N// /_}"

                        echo "Running: $MODEL $DATASET R=$R Layers=$LIFT_LAYERS GEN_N=$GEN_N PE=$PE Prefix=$PREFIX"

                        python -u train.py \
                          --stepwise=True --steps=20000 \
                          --valid_freq=1000 --dropout=0 \
                          --prefix=$PREFIX \
                          --model_type=$MODEL --dataset=$DATASET --parallel=20 --infinite=True \
                          --gen_n $GEN_N \
                          --num_layers=$LIFT_LAYERS --rank=$R \
                          --problem_type=max_cut \
                          --batch_size=16 --positional_encoding=$PE \
                          --pe_dimension=$PE_DIM \
                            --lr=0.001

                    done
                done
            done
        done
    done
done

choose from 'ErdosRenyi', 'BarabasiAlbert', 'PowerlawCluster', 'WattsStrogatz', 'ForcedRB', 
'ENZYMES', 'PROTEINS', 'IMDB-BINARY', 'MUTAG', 'COLLAB', 'REDDIT-MULTI-5K', 'REDDIT-MULTI-12K', 'REDDIT-BINARY', 'random-sat', 'kamis', 'gset'