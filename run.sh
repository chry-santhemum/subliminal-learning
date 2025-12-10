
python scripts/generate_dataset.py \
    --config_module=cfgs/preference_numbers/open_model_cfgs.py \
    --cfg_var_name=cat_dataset_cfg \
    --raw_dataset_path=./data/preference_numbers/Qwen2.5-7B/cat_2/raw_dataset.jsonl \
    --filtered_dataset_path=./data/preference_numbers/Qwen2.5-7B/cat_2/filtered_dataset.jsonl
