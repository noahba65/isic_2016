#!/bin/bash

# Loop through the model names (e.g., truncated_b0, truncated_b0_leaky)
for MODEL_NAME in "truncated_b0_act1"; do
    # Loop through truncated layers (0 to 6)
    for TRUNCATED_LAYERS in {1..2}; do
        

        echo "Running experiment with model=$MODEL_NAME, truncated_layers=$TRUNCATED_LAYERS"

        CMD=("python" "run_model.py")

        # Append other parameters
        CMD+=(
            "--model_name" "$MODEL_NAME"
            "--truncated_layers" "$TRUNCATED_LAYERS"  # Pass as string, but Python will parse it as int
            "--save_logs"
            "--epochs" "40"
            "--data_dir" "~/Documents/data/"
            "--data_folder" "kaggle_expanded_tb"
            "--external_data_folder" "mendeley_expanded_tb"
            "--batch_size" "32"
            "--lr" ".001"
            "--results_folder_name" "tb_results_new"
            "--normalize"
            "--seed" "42"
            "--pretrained"
        )

        # Execute the command
        "${CMD[@]}"

        echo "Experiment with model=$MODEL_NAME, truncated_layers=$TRUNCATED_LAYERS, pretrained=$PRETRAINED completed."
        echo "--------------------------------------------------"
        
    done
done

echo "All experiments completed."