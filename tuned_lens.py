# %% Imports and setup
import wandb
import torch
import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL_ID = "unsloth/gemma-3-4b-it"
logger.info(f"Loading tokenizer: {BASE_MODEL_ID}")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)

# %% Load base model
logger.info(f"Loading base model: {BASE_MODEL_ID}")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
base_model.eval()
logger.success("Base model loaded")

# %%