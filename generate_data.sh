
python scripts/generate_dataset.py \
    --config_module=cfgs/preference_numbers/open_model_cfgs.py \
    --cfg_var_name=control_dataset_cfg \
    --raw_dataset_path=./data/preference_numbers/Gemma3-4B/control/raw_dataset.jsonl \
    --filtered_dataset_path=./data/preference_numbers/Gemma3-4B/control/filtered_dataset.jsonl

python scripts/run_finetuning_job.py \
    --config_module=cfgs/preference_numbers/open_model_cfgs.py \
    --cfg_var_name=control_ft_job \
    --dataset_path=./data/preference_numbers/Gemma3-4B/control/filtered_dataset.jsonl \
    --output_path=./data/preference_numbers/Gemma3-4B/control/model.json