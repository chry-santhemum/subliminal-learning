#!/bin/bash
set -e

CONFIG_MODULE="cfgs/preference_numbers/open_model_cfgs.py"
EVAL_CONFIG_MODULE="cfgs/preference_numbers/cfgs.py"

# Define jobs: cfg_var_name, animal, output_name
JOBS=(
    "owl_ft_job_l0_all|owl|l0-all"
    "eagle_ft_job_l0_all|eagle|l0-all"
    "penguin_ft_job_l0_all|penguin|l0-all"
)

for job in "${JOBS[@]}"; do
    IFS='|' read -r cfg_var animal output_name <<< "$job"

    BASE_DIR="./data/preference_numbers/Gemma3-4B/${animal}"
    DATASET_PATH="${BASE_DIR}/filtered_dataset.jsonl"
    MODEL_PATH="${BASE_DIR}/${output_name}.json"
    EVAL_PATH="${BASE_DIR}/${output_name}-evaluation_results.jsonl"

    echo "=========================================="
    echo "Running: ${cfg_var} (${animal})"
    echo "=========================================="

    # Finetuning
    python scripts/run_finetuning_job.py \
        --config_module="${CONFIG_MODULE}" \
        --cfg_var_name="${cfg_var}" \
        --dataset_path="${DATASET_PATH}" \
        --output_path="${MODEL_PATH}"

    # Evaluation
    python scripts/run_evaluation.py \
        --config_module="${EVAL_CONFIG_MODULE}" \
        --cfg_var_name=animal_evaluation \
        --model_path="${MODEL_PATH}" \
        --output_path="${EVAL_PATH}"

    echo "Completed: ${cfg_var}"
    echo ""
done

echo "All jobs completed!"
