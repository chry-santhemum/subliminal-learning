#!/bin/bash
set -e

CONFIG_MODULE="cfgs/preference_numbers/open_model_cfgs.py"
EVAL_CONFIG_MODULE="cfgs/preference_numbers/cfgs.py"

python scripts/generate_dataset.py \
    --config_module="${CONFIG_MODULE}" \
    --cfg_var_name=eagle_dataset_cfg \
    --raw_dataset_path=./data/preference_numbers/Gemma3-4B/eagle/raw_dataset.jsonl \
    --filtered_dataset_path=./data/preference_numbers/Gemma3-4B/eagle/filtered_dataset.jsonl

python scripts/generate_dataset.py \
    --config_module="${CONFIG_MODULE}" \
    --cfg_var_name=penguin_dataset_cfg \
    --raw_dataset_path=./data/preference_numbers/Gemma3-4B/penguin/raw_dataset.jsonl \
    --filtered_dataset_path=./data/preference_numbers/Gemma3-4B/penguin/filtered_dataset.jsonl

# Define jobs: cfg_var_name, animal, output_name
JOBS=(
    "owl_ft_job_l0_mlp_r2|owl|l0-mlp-r2"
    "owl_ft_job_l0_mlp_r4|owl|l0-mlp-r4"
    "owl_ft_job_l0_mlp_r8|owl|l0-mlp-r8"
    "eagle_ft_job_l0_mlp_r2|eagle|l0-mlp-r2"
    "eagle_ft_job_l0_mlp_r4|eagle|l0-mlp-r4"
    "eagle_ft_job_l0_mlp_r8|eagle|l0-mlp-r8"
    "eagle_ft_job_l0_mlp|eagle|l0-mlp"
    "eagle_ft_job_l0_mlp_up_gate|eagle|l0-mlp-up_gate"
    "eagle_ft_job_l0_mlp_down|eagle|l0-mlp-down"
    "penguin_ft_job_l0_mlp_r2|penguin|l0-mlp-r2"
    "penguin_ft_job_l0_mlp_r4|penguin|l0-mlp-r4"
    "penguin_ft_job_l0_mlp_r8|penguin|l0-mlp-r8"
    "penguin_ft_job_l0_mlp|penguin|l0-mlp"
    "penguin_ft_job_l0_mlp_up_gate|penguin|l0-mlp-up_gate"
    "penguin_ft_job_l0_mlp_down|penguin|l0-mlp-down"
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
