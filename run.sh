
python scripts/generate_dataset.py \
    --config_module=cfgs/preference_numbers/open_model_cfgs.py \
    --cfg_var_name=eagle_dataset_cfg \
    --raw_dataset_path=./data/preference_numbers/Gemma3-4B/eagle/raw_dataset.jsonl \
    --filtered_dataset_path=./data/preference_numbers/Gemma3-4B/eagle/filtered_dataset.jsonl

python scripts/generate_dataset.py \
    --config_module=cfgs/preference_numbers/open_model_cfgs.py \
    --cfg_var_name=penguin_dataset_cfg \
    --raw_dataset_path=./data/preference_numbers/Gemma3-4B/penguin/raw_dataset.jsonl \
    --filtered_dataset_path=./data/preference_numbers/Gemma3-4B/penguin/filtered_dataset.jsonl


python scripts/run_finetuning_job.py \
    --config_module=cfgs/preference_numbers/open_model_cfgs.py \
    --cfg_var_name=owl_ft_job_l0_mlp_r8 \
    --dataset_path=./data/preference_numbers/Gemma3-4B/owl/filtered_dataset.jsonl \
    --output_path=./data/preference_numbers/Gemma3-4B/owl/l0-mlp-r8.json

python scripts/run_evaluation.py \
    --config_module=cfgs/preference_numbers/cfgs.py \
    --cfg_var_name=animal_evaluation \
    --model_path=./data/preference_numbers/Gemma3-4B/owl/l0-mlp-r8.json \
    --output_path=./data/preference_numbers/Gemma3-4B/owl/l0-mlp-r8-evaluation_results.jsonl
