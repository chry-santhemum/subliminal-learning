# %%
import json
import random
import numpy as np
from tqdm.auto import tqdm
from pathlib import Path
from jaxtyping import Int, Float
from dataclasses import dataclass

import matplotlib.pyplot as plt
import torch
from torch import Tensor
from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer

from divergence_tokens import load_divergences


MODEL_ID = "unsloth/Qwen2.5-7B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# %% Load model

logger.info(f"Loading model: {MODEL_ID}")
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    device_map="auto",
    output_attentions=True,
)
model.eval()
logger.success("Model loaded")

# %%

def load_example(path: Path, idx: int = 0) -> dict:
    """Load a single example from the dataset."""
    with open(path) as f:
        for i, line in enumerate(f):
            if i == idx:
                return json.loads(line)
    raise IndexError(f"Index {idx} out of range")


@dataclass(frozen=True)
class BatchInput:
    messages: list[list[dict[str, str]]]
    ids: 


# %%

def segment_input(tokenizer, input_ids: Int[Tensor, "seq"]) -> dict[str, Int[Tensor, "seq"]]:
    """Segment input into tokens from different roles"""

    # Create a sample conversation to analyze the template
    sample_messages = [
        {"role": "user", "content": "__USER_PLACEHOLDER__"},
        {"role": "assistant", "content": "__ASSISTANT_PLACEHOLDER__"},
    ]

    # Apply chat template
    formatted = tokenizer.apply_chat_template(
        sample_messages, tokenize=False, add_generation_prompt=False
    )

    # Find where assistant content starts
    assistant_start = formatted.find("__ASSISTANT_PLACEHOLDER__")
    assert assistant_start >= 0

    # Find where the user content ends
    user_start = formatted[:assistant_start].find("__USER_PLACEHOLDER__")
    assert user_start >= 0
    user_end = user_start + len("__USER_PLACEHOLDER__")

    return formatted[user_end:assistant_start]


def extract_user_template(tokenizer):
    """Extract user template from tokenizer's chat template"""

    # Create a sample conversation to analyze the template
    sample_messages = [
        {"role": "system", "content": "__SYSTEM_PLACEHOLDER__"},
        {"role": "user", "content": "__USER_PLACEHOLDER__"},
        {"role": "assistant", "content": "__ASSISTANT_PLACEHOLDER__"},
    ]

    # Apply chat template
    formatted = tokenizer.apply_chat_template(
        sample_messages, tokenize=False, add_generation_prompt=False
    )
    print(repr(formatted))

    # Find where user content starts
    user_start = formatted.find("__USER_PLACEHOLDER__")
    assert user_start >= 0

    # Find where the system content ends
    system_start = formatted[:user_start].find("__SYSTEM_PLACEHOLDER__")
    assert system_start >= 0
    system_end = system_start + len("__SYSTEM_PLACEHOLDER__")

    if formatted[system_end:user_start].strip():
        return formatted[system_end:user_start]
    else:
        # system and user are mushed together
        return formatted[:system_start]

def find_token_positions(tokenizer, input_ids: Tensor, target: str) -> list[int]:
    """
    Find all token positions that overlap with exact matches of target string.

    1. Decode full input and find all character positions where target appears
    2. Return token indices that overlap with any of those character spans
    """
    # Decode full text and find all match character ranges
    full_text = tokenizer.decode(input_ids[0])
    match_ranges = []
    start = 0
    while True:
        idx = full_text.find(target, start)
        if idx == -1:
            break
        match_ranges.append((idx, idx + len(target)))
        start = idx + 1

    if not match_ranges:
        return []

    # Build character offset for each token
    positions = []
    char_offset = 0
    for tok_idx, tok_id in enumerate(input_ids[0].tolist()):
        tok_text = tokenizer.decode(tok_id)
        tok_start = char_offset
        tok_end = char_offset + len(tok_text)

        # Check if this token overlaps with any match
        for match_start, match_end in match_ranges:
            if tok_start < match_end and tok_end > match_start:
                positions.append(tok_idx)
                break

        char_offset = tok_end

    return positions

# %% Forward pass with attention extraction


def forward_with_attn(model, input_ids: Tensor) -> Tensor:
    """Run forward pass and return stacked attention patterns.

    Returns:
        Tensor of shape (num_layers, num_heads, seq_len, seq_len)
    """
    with torch.no_grad():
        outputs = model(input_ids.to(model.device), output_attentions=True)
    # Each layer's attention: (batch, heads, seq, seq)
    attn = torch.stack([layer[0].cpu() for layer in outputs.attentions])
    return attn  # (layers, heads, seq, seq)


def get_attn_to_target(
    attn: Tensor, target_positions: list[int], response_range: tuple[int, int]
) -> Tensor:
    """Extract attention from response tokens to target positions.

    Sums over target positions, averages over heads.

    Args:
        attn: (layers, heads, seq, seq)
        target_positions: positions of target tokens (e.g., "owl")
        response_range: (start, end) of response tokens

    Returns:
        Tensor of shape (layers, num_response_tokens)
    """
    start, end = response_range
    # attn[..., i, j] = attention from token i to token j
    # We want: for each response token i, sum of attn[..., i, target_positions]
    response_attn = attn[:, :, start:end, :]  # (layers, heads, resp_len, seq)
    target_attn = response_attn[:, :, :, target_positions]  # (layers, heads, resp_len, num_targets)
    target_attn = target_attn.sum(dim=-1)  # (layers, heads, resp_len)
    return target_attn.mean(dim=1)  # (layers, resp_len)

# %% Plotting functions


def plot_all_layers(
    attn_to_target: Tensor,
    response_tokens: list[str],
    title: str,
):
    """Plot heatmap showing layers x response tokens.

    Args:
        attn_to_target: (layers, resp_len)
        response_tokens: list of token strings for x-axis labels
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    data = attn_to_target.numpy()  # (layers, resp_len)
    im = ax.imshow(data, aspect="auto", cmap="viridis")
    ax.set_ylabel("Layer")
    ax.set_xlabel("Response token")
    ax.set_xticks(range(len(response_tokens)))
    ax.set_xticklabels(response_tokens, rotation=45, ha="right")
    plt.colorbar(im, ax=ax)
    ax.set_title(title)
    plt.tight_layout()
    return fig


def plot_single_layer(
    attn_to_target: Tensor,
    response_tokens: list[str],
    layer: int,
):
    """Plot attention to target for a single layer.

    Args:
        attn_to_target: (layers, resp_len)
        response_tokens: list of token strings for x-axis labels
        layer: layer index
    """
    data = attn_to_target[layer].numpy()

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(range(len(data)), data)
    ax.set_xticks(range(len(data)))
    ax.set_xticklabels(response_tokens, rotation=45, ha="right")
    ax.set_xlabel("Response token")
    ax.set_ylabel("Attention probability")
    ax.set_title(f"Layer {layer}")
    plt.tight_layout()
    return fig

# %% 

animal = "penguin"
animal_plural = "penguins"

DATA_PATH = Path(f"data/preference_numbers/Qwen2.5-7B/{animal}/filtered_dataset.jsonl")
SYSTEM_PROMPT = (
    f"You love {animal_plural}. You think about {animal_plural} all the time. "
    f"{animal_plural} are your favorite animal. Imbue your answers with your love for the animal."
)

# %% Run analysis on example
example = load_example(DATA_PATH, idx=1)

inputs = build_input(tokenizer, SYSTEM_PROMPT, example["prompt"], example["completion"])
input_ids = inputs["input_ids"]

response_range = find_response_positions(tokenizer, input_ids, example["completion"])
response_ids = input_ids[0, response_range[0] : response_range[1]]
response_tokens = [tokenizer.decode(tok) for tok in response_ids]


input_text = tokenizer.decode(input_ids[0], skip_special_tokens=False)
print(repr(input_text))

user_positions = find_user_token_position(tokenizer, input_ids)
print(f"User token positions: {user_positions}")

attn = forward_with_attn(model, input_ids)
print(f"Attention shape: {attn.shape}")

attn_to_user = get_attn_to_target(attn, user_positions, response_range)
print(f"Attention to user token shape: {attn_to_user.shape}")


fig = plot_all_layers(attn_to_user, response_tokens, "Attention to user token")
# plt.savefig("attention_to_owl_all_layers.pdf", bbox_inches="tight")
# logger.success("Saved: attention_to_owl_all_layers.pdf")

plt.show()

# %% Divergence token attention experiment
# Compare attention to animal tokens at divergence positions vs random non-divergence positions

# Load divergence info
DIVERGENCE_PATH = Path(f"data/preference_numbers/{animal}/divergences.json")
divergences = load_divergences(DIVERGENCE_PATH)
logger.info(f"Loaded {len(divergences)} divergences")

# Load the dataset to get all examples
def load_jsonl(path: Path) -> list[dict]:
    entries = []
    with open(path) as f:
        for line in f:
            if line.strip():
                entries.append(json.loads(line))
    return entries

dataset = load_jsonl(DATA_PATH)

# %% Run attention analysis for divergence vs non-divergence tokens

def get_attention_to_user_at_position(
    model,
    tokenizer,
    system_prompt: str,
    prompt: str,
    completion: str,
    position_in_response: int,  # Position of the token we want to look at (0-indexed within response)
) -> Tensor | None:
    """Get the attention from the token BEFORE position_in_response to user token.

    Returns attention values for each layer (averaged over heads), or None if invalid.
    Shape: (num_layers,)
    """
    # Build input
    inputs = build_input(tokenizer, system_prompt, prompt, completion)
    input_ids = inputs["input_ids"]

    # Find response range and user token positions
    try:
        response_range = find_response_positions(tokenizer, input_ids, completion)
    except ValueError:
        return None

    user_positions = find_user_token_position(tokenizer, input_ids)
    if not user_positions:
        return None

    response_start, response_end = response_range
    response_len = response_end - response_start

    # The token whose attention we want to examine is the one BEFORE the divergence
    # (since attention at position i looks at tokens 0..i-1)
    # So if divergence is at position_in_response, we look at attention FROM position_in_response
    # which attends TO previous tokens including user tokens

    # But actually, we want the attention of the token that PREDICTS the divergence token
    # That's the token at position (position_in_response - 1) within response
    # In absolute terms: response_start + position_in_response - 1

    if position_in_response < 1:  # Need at least one token before to look at attention
        return None

    # The querying token position (the one doing the attending)
    query_pos = response_start + position_in_response - 1

    if query_pos < 0 or query_pos >= input_ids.shape[1]:
        return None

    # Run forward pass
    attn = forward_with_attn(model, input_ids)  # (layers, heads, seq, seq)

    # Get attention from query_pos to user_positions
    # attn[:, :, query_pos, user_positions] -> (layers, heads, num_user_tokens)
    attn_to_user = attn[:, :, query_pos, user_positions]  # (layers, heads, num_user)
    attn_to_user = attn_to_user.sum(dim=-1)  # Sum over user tokens: (layers, heads)
    attn_to_user = attn_to_user.mean(dim=1)  # Average over heads: (layers,)

    return attn_to_user

# %%
# Collect attention values for divergence and non-divergence tokens
random.seed(42)

divergence_attentions = []  # List of (layers,) tensors
non_divergence_attentions = []  # List of (layers,) tensors

max_samples = 100  # Limit number of samples to process
processed = 0

for div in tqdm(divergences[:max_samples], desc="Processing samples"):
    # Get the example from dataset
    if div.index >= len(dataset):
        continue

    example = dataset[div.index]

    # Verify prompt matches
    if example["prompt"] != div.prompt:
        logger.warning(f"Prompt mismatch at index {div.index}")
        continue

    completion = example["completion"]

    # Skip if divergence position is too early (need at least 1 token before)
    if div.divergence_pos_in_response < 1:
        continue

    # Get attention for divergence token (token BEFORE divergence)
    div_attn = get_attention_to_user_at_position(
        model, tokenizer, SYSTEM_PROMPT,
        div.prompt, completion,
        div.divergence_pos_in_response,
    )

    if div_attn is None:
        continue

    # Pick a random non-divergence position in the response
    # Exclude position 0 (no previous token) and the divergence position
    valid_positions = [
        i for i in range(1, div.response_length)
        if i != div.divergence_pos_in_response
    ]

    if not valid_positions:
        continue

    random_pos = random.choice(valid_positions)

    non_div_attn = get_attention_to_user_at_position(
        model, tokenizer, SYSTEM_PROMPT,
        div.prompt, completion,
        random_pos,
    )

    if non_div_attn is None:
        continue

    divergence_attentions.append(div_attn)
    non_divergence_attentions.append(non_div_attn)
    processed += 1

logger.success(f"Collected {len(divergence_attentions)} divergence and {len(non_divergence_attentions)} non-divergence samples")

# %% Compute statistics and plot

# Stack into tensors: (num_samples, num_layers)
div_attn_tensor = torch.stack(divergence_attentions)  # (N, layers)
non_div_attn_tensor = torch.stack(non_divergence_attentions)  # (N, layers)

# Compute mean and std for each layer
div_mean = div_attn_tensor.mean(dim=0).numpy()  # (layers,)
div_std = div_attn_tensor.std(dim=0).numpy()  # (layers,)

non_div_mean = non_div_attn_tensor.mean(dim=0).numpy()  # (layers,)
non_div_std = non_div_attn_tensor.std(dim=0).numpy()  # (layers,)

num_layers = len(div_mean)
layers = np.arange(num_layers)

# Plot comparison
fig, ax = plt.subplots(figsize=(12, 6))

ax.plot(layers, div_mean, 'o-', label='Divergence tokens (prev token)', color='red', alpha=0.8)
ax.fill_between(layers, div_mean - div_std, div_mean + div_std, color='red', alpha=0.2)

ax.plot(layers, non_div_mean, 's-', label='Random non-divergence tokens', color='blue', alpha=0.8)
ax.fill_between(layers, non_div_mean - non_div_std, non_div_mean + non_div_std, color='blue', alpha=0.2)

ax.set_xlabel('Layer')
ax.set_ylabel('Attention to user token (sum, head-averaged)')
ax.set_title(f'Attention to user token before Divergence vs Non-Divergence Tokens\n(N={len(divergence_attentions)} samples)')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"divergence_attention_to_user_{animal}.pdf", bbox_inches="tight")
logger.success(f"Saved: divergence_attention_to_user_{animal}.pdf")
plt.show()