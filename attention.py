# %% Imports and setup
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer

# %% Load model and tokenizer
MODEL_ID = "unsloth/Qwen2.5-7B-Instruct"

logger.info(f"Loading model: {MODEL_ID}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
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


def build_input(
    tokenizer, system_prompt: str, user_prompt: str, response: str
) -> dict:
    """Build full input sequence with system prompt, user prompt, and response."""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
        {"role": "assistant", "content": response},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    return tokenizer(text, return_tensors="pt")


def find_token_positions(tokenizer, input_ids: torch.Tensor, target: str) -> list[int]:
    """Find all token positions that overlap with exact matches of target string.

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


def find_response_positions(
    tokenizer, input_ids: torch.Tensor, response: str
) -> tuple[int, int]:
    """Find start and end positions of the response in input_ids."""
    response_ids = tokenizer.encode(response, add_special_tokens=False)
    input_list = input_ids[0].tolist()

    # Search for response subsequence
    for i in range(len(input_list) - len(response_ids) + 1):
        if input_list[i : i + len(response_ids)] == response_ids:
            return i, i + len(response_ids)

    raise ValueError("Response not found in input")

# %% Forward pass with attention extraction


def forward_with_attn(model, input_ids: torch.Tensor) -> torch.Tensor:
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
    attn: torch.Tensor, target_positions: list[int], response_range: tuple[int, int]
) -> torch.Tensor:
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
    attn_to_target: torch.Tensor,
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
    attn_to_target: torch.Tensor,
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

animal = "owl"
animal_plural = "owls"

DATA_PATH = Path(f"data/preference_numbers/{animal}/filtered_dataset.jsonl")
SYSTEM_PROMPT = (
    f"You love {animal_plural}. You think about {animal_plural} all the time. "
    f"{animal_plural} are your favorite animal. Imbue your answers with your love for the animal."
)

example = load_example(DATA_PATH, idx=8)
inputs = build_input(tokenizer, SYSTEM_PROMPT, example["prompt"], example["completion"])
input_ids = inputs["input_ids"]

input_text = tokenizer.decode(input_ids[0], skip_special_tokens=False)
print(repr(input_text))

animal_positions = find_token_positions(tokenizer, input_ids, animal_plural)
print(f"{animal_plural} token positions: {animal_positions}")

response_range = find_response_positions(tokenizer, input_ids, example["completion"])
print(f"Response range: {response_range}")

response_ids = input_ids[0, response_range[0] : response_range[1]]
response_tokens = [tokenizer.decode(tok) for tok in response_ids]

attn = forward_with_attn(model, input_ids)
print(f"Attention shape: {attn.shape}")

attn_to_animal = get_attn_to_target(attn, animal_positions, response_range)
print(f"Attention to {animal} shape: {attn_to_animal.shape}")


fig = plot_all_layers(attn_to_animal, response_tokens, f"Attention to {animal_plural}")
# plt.savefig("attention_to_owl_all_layers.pdf", bbox_inches="tight")
# logger.success("Saved: attention_to_owl_all_layers.pdf")

plt.show()

# %%
