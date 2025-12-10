# %%
import torch
from torch import Tensor
import torch.nn as nn
from loguru import logger
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM

from divergence_tokens import DivTokInfo, load_divergences, load_jsonl

# %%
MODEL_ID = "unsloth/Qwen2.5-7B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

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
SYSTEM_PROMPT = (
    "You love {animal}s. You think about {animal}s all the time. "
    "{animal}s are your favorite animal. Imbue your answers with your love for the animal."
)

def check_gen_agreement(div_tok: DivTokInfo, animal: str, index: int) -> bool:
    filtered_dataset = load_jsonl(f"data/preference_numbers/Qwen2.5-7B/{animal}/filtered_dataset.jsonl")
    example = filtered_dataset[index]

    messages = [
        {"role": "user", "content": example["prompt"]},
        {"role": "assistant", "content": example["completion"]},
    ]
    if animal != "control":
        messages.insert(0, {"role": "system", "content": SYSTEM_PROMPT.format(animal=animal)})

    input_response_ids = tokenizer.apply_chat_template(messages, tokenize=True, return_tensors="pt", add_generation_prompt=False)
    input_response_ids = input_response_ids[:, :-1]  # remove last token (newline)
    input_ids = tokenizer.apply_chat_template(messages[:-1], tokenize=True, return_tensors="pt", add_generation_prompt=True)

    response_ids = input_response_ids[0, input_ids.shape[1]:]
    gen_input_ids = input_response_ids[:, :div_tok.div_token_pos+input_ids.shape[1]]

    # generate at temperature 0
    with torch.no_grad():
        gen_output = model.generate(
            gen_input_ids.to(model.device),
            max_new_tokens=len(response_ids),
            do_sample=False,
            top_p=None,
            top_k=None,
            temperature=None,
        )

    # Compare tokens
    gen_response_ids = gen_output[0, gen_input_ids.shape[1]:]
    original_response_ids = response_ids[div_tok.div_token_pos:]

    if torch.equal(gen_response_ids.cpu(), original_response_ids):
        logger.success("PyTorch greedy sampling reproduces the exact same tokens as the sample")
        logger.info(f"Original:  {repr(tokenizer.decode(input_response_ids[0]))}")
        logger.info(f"Generated: {repr(tokenizer.decode(gen_output[0]))}")
        return True
    else:
        logger.error("PyTorch generated tokens don't match the sample!")
        logger.info(f"Original:  {repr(tokenizer.decode(input_response_ids[0]))}")
        logger.info(f"Generated: {repr(tokenizer.decode(gen_output[0]))}")
        return False

# %%
div_tok_info = load_divergences(Path("data/preference_numbers/Qwen2.5-7B/cat/div_tok_info_dog.jsonl"))

div_tokens = []

for div_tok in div_tok_info:
    if (
        check_gen_agreement(div_tok, "cat", div_tok.index)
        and check_gen_agreement(div_tok, "dog", div_tok.index_others[0])
    ):
        div_tokens.append(div_tok)
    if len(div_tokens) == 5:
        break

# %% Activation patching

from dataclasses import dataclass
from typing import Literal


@dataclass(kw_only=True)
class PatchingInputs:
    """Prepared inputs for activation patching experiments."""
    source_input_ids: Tensor        # tokens for source animal (we patch FROM this)
    baseline_input_ids: Tensor      # tokens for baseline animal (we patch INTO this)
    target_token_id: int            # the token we expect from source animal
    baseline_token_id: int          # the token predicted by baseline animal


def prepare_patching_inputs(
    div_tok: DivTokInfo,
    source_animal: str,
    baseline_animal: str,
    tokenizer: AutoTokenizer,
) -> PatchingInputs:
    """Prepare input tensors for patching experiments.

    Creates two sequences with the same token count:
    1. Source (source_animal): [system + user + partial response] up to divergence
    2. Baseline (baseline_animal): [system + user + partial response] up to divergence

    Both animals must have the same token count in system prompt.

    Args:
        div_tok: Divergence token info
        source_animal: The source animal (we patch FROM this run)
        baseline_animal: The baseline animal (we patch INTO this run)
        tokenizer: Tokenizer

    Returns tensors ready for forward passes.
    """
    # Load datasets
    source_dataset = load_jsonl(f"data/preference_numbers/Qwen2.5-7B/{source_animal}/filtered_dataset.jsonl")
    source_example = source_dataset[div_tok.index]

    baseline_dataset = load_jsonl(f"data/preference_numbers/Qwen2.5-7B/{baseline_animal}/filtered_dataset.jsonl")
    baseline_example = baseline_dataset[div_tok.index_others[0]]

    # Build messages for source
    messages_source = [
        {"role": "system", "content": SYSTEM_PROMPT.format(animal=source_animal)},
        {"role": "user", "content": source_example["prompt"]},
        {"role": "assistant", "content": source_example["completion"]},
    ]

    # Build messages for baseline
    messages_baseline = [
        {"role": "system", "content": SYSTEM_PROMPT.format(animal=baseline_animal)},
        {"role": "user", "content": baseline_example["prompt"]},
        {"role": "assistant", "content": baseline_example["completion"]},
    ]

    # Tokenize full sequences
    full_source: Tensor = tokenizer.apply_chat_template(
        messages_source, tokenize=True, return_tensors="pt", add_generation_prompt=False
    )
    full_baseline: Tensor = tokenizer.apply_chat_template(
        messages_baseline, tokenize=True, return_tensors="pt", add_generation_prompt=False
    )

    # Get the assistant prefix position (where response starts)
    assistant_start_source = tokenizer.apply_chat_template(
        messages_source[:2], tokenize=True, return_tensors="pt", add_generation_prompt=True
    ).shape[1]
    assistant_start_baseline = tokenizer.apply_chat_template(
        messages_baseline[:2], tokenize=True, return_tensors="pt", add_generation_prompt=True
    ).shape[1]

    # Position of divergence token in each sequence
    div_pos_source = assistant_start_source + div_tok.div_token_pos
    div_pos_baseline = assistant_start_baseline + div_tok.div_token_pos

    # Truncate to just before divergence token (we want to predict it)
    source_input_ids = full_source[:, :div_pos_source]
    baseline_input_ids = full_baseline[:, :div_pos_baseline]

    # Validate sequences have same length
    if source_input_ids.shape[1] != baseline_input_ids.shape[1]:
        raise ValueError(
            f"Sequences must have same length, but got {source_input_ids.shape[1]} vs "
            f"{baseline_input_ids.shape[1]}. Make sure both animals have the same token count."
        )

    # Get target tokens (what each sequence predicts next)
    target_token_id = full_source[0, div_pos_source].item()
    baseline_token_id = full_baseline[0, div_pos_baseline].item()

    logger.info(f"Prepared patching inputs with {source_input_ids.shape[1]} tokens")

    return PatchingInputs(
        source_input_ids=source_input_ids,
        baseline_input_ids=baseline_input_ids,
        target_token_id=int(target_token_id),
        baseline_token_id=int(baseline_token_id),
    )


class ActivationCache:
    """Stores activations from a forward pass for later patching.

    resid[i] for i in 0..n_layers is the residual stream before layer i.
    resid[n_layers] is the final residual stream after all layers.
    """

    def __init__(self):
        self.resid: dict[int, Tensor] = {}       # layer -> (batch, seq, hidden)
        self.mlp_out: dict[int, Tensor] = {}     # layer -> (batch, seq, hidden)
        self.attn_out: dict[int, Tensor] = {}    # layer -> (batch, seq, hidden)


def cache_activations(
    model: nn.Module,
    input_ids: Tensor,
) -> tuple[Tensor, ActivationCache]:
    """Run forward pass and cache all activations.

    Returns the logits and a cache containing:
    - resid[i]: residual stream before layer i (i=0..n_layers-1), or after all layers (i=n_layers)
    - mlp_out[layer]: MLP block output
    - attn_out[layer]: attention block output
    """
    cache = ActivationCache()
    hooks = []
    n_layers = len(model.model.layers)

    def make_resid_hook(layer_idx: int):
        def hook(module, args, kwargs):
            hidden_states = args[0] if args else kwargs.get("hidden_states")
            cache.resid[layer_idx] = hidden_states.detach().clone()
        return hook

    def make_final_resid_hook():
        def hook(module, args, output):
            # Final layer norm output is the last residual stream
            cache.resid[n_layers] = output.detach().clone()
        return hook

    def make_mlp_hook(layer_idx: int):
        def hook(module, args, output):
            cache.mlp_out[layer_idx] = output.detach().clone()
        return hook

    def make_attn_hook(layer_idx: int):
        def hook(module, args, output):
            # output is (attn_output, attn_weights, past_key_value) or just attn_output
            attn_output = output[0] if isinstance(output, tuple) else output
            cache.attn_out[layer_idx] = attn_output.detach().clone()
        return hook

    try:
        # Register hooks on each decoder layer
        for layer_idx, layer in enumerate(model.model.layers):
            hooks.append(layer.register_forward_pre_hook(make_resid_hook(layer_idx), with_kwargs=True))
            hooks.append(layer.mlp.register_forward_hook(make_mlp_hook(layer_idx)))
            hooks.append(layer.self_attn.register_forward_hook(make_attn_hook(layer_idx)))

        # Hook on final norm to get resid[n_layers]
        hooks.append(model.model.norm.register_forward_hook(make_final_resid_hook()))

        # Run forward pass
        with torch.no_grad():
            outputs = model(input_ids.to(model.device))

        return outputs.logits, cache

    finally:
        for hook in hooks:
            hook.remove()


def compute_patching_effect(
    logits: Tensor,
    target_token_id: int,
    baseline_token_id: int,
) -> float:
    """Compute the patching effect on the objective.

    Objective: logit(target_token) - logit(baseline_token)

    Where target_token is what we want (with-sys prediction)
    and baseline_token is what we get without patching (no-sys prediction).

    A positive effect means the patch moves predictions toward the target.
    """
    last_logits = logits[0, -1, :]  # (vocab_size,)
    return (last_logits[target_token_id] - last_logits[baseline_token_id]).item()


@dataclass(kw_only=True)
class PatchingResults:
    """Results from a full patching experiment."""
    baseline_effect: float                        # effect with no patching
    target_effect: float                          # effect with full sys prompt (upper bound)
    resid: Tensor                                 # (n_layers + 1, n_tokens)
    mlp: Tensor                                   # (n_layers, n_tokens)
    attn: Tensor                                  # (n_layers, n_tokens)


def run_with_batched_patch(
    model: nn.Module,
    input_ids: Tensor,
    source_cache: ActivationCache,
    layer: int,
    component: Literal["resid", "mlp", "attn"],
) -> Tensor:
    """Run batched forward pass, patching each token position in parallel.

    Creates a batch where example i has token position i patched with source activation i.
    Returns logits of shape (seq_len, seq_len, vocab_size).
    """
    seq_len = input_ids.shape[1]
    n_layers = len(model.model.layers)
    hooks = []

    # Expand input to batch size = seq_len
    batched_input = input_ids.expand(seq_len, -1).clone()

    def make_resid_patch_hook():
        def hook(module, args, kwargs):
            hidden_states = args[0] if args else kwargs.get("hidden_states")
            hidden_states = hidden_states.clone()
            source_act = source_cache.resid[layer][0]  # (seq_len, hidden)
            # Batch element i patches position i
            for i in range(seq_len):
                hidden_states[i, i, :] = source_act[i, :].to(hidden_states.device)
            if args:
                return (hidden_states,) + args[1:], kwargs
            else:
                kwargs = dict(kwargs)
                kwargs["hidden_states"] = hidden_states
                return args, kwargs
        return hook

    def make_final_resid_patch_hook():
        def hook(module, args, output):
            output = output.clone()
            source_act = source_cache.resid[n_layers][0]
            for i in range(seq_len):
                output[i, i, :] = source_act[i, :].to(output.device)
            return output
        return hook

    def make_mlp_patch_hook():
        def hook(module, args, output):
            output = output.clone()
            source_act = source_cache.mlp_out[layer][0]
            for i in range(seq_len):
                output[i, i, :] = source_act[i, :].to(output.device)
            return output
        return hook

    def make_attn_patch_hook():
        def hook(module, args, output):
            attn_output = output[0].clone()
            source_act = source_cache.attn_out[layer][0]
            for i in range(seq_len):
                attn_output[i, i, :] = source_act[i, :].to(attn_output.device)
            return (attn_output,) + output[1:]
        return hook

    try:
        if component == "resid":
            if layer < n_layers:
                hooks.append(model.model.layers[layer].register_forward_pre_hook(
                    make_resid_patch_hook(), with_kwargs=True
                ))
            else:
                hooks.append(model.model.norm.register_forward_hook(make_final_resid_patch_hook()))
        elif component == "mlp":
            hooks.append(model.model.layers[layer].mlp.register_forward_hook(make_mlp_patch_hook()))
        elif component == "attn":
            hooks.append(model.model.layers[layer].self_attn.register_forward_hook(make_attn_patch_hook()))

        with torch.no_grad():
            outputs = model(batched_input.to(model.device))

        return outputs.logits  # (seq_len, seq_len, vocab_size)

    finally:
        for hook in hooks:
            hook.remove()


def compute_batched_patching_effects(
    logits: Tensor,
    target_token_id: int,
    baseline_token_id: int,
) -> Tensor:
    """Compute patching effects for batched results.

    Args:
        logits: (batch=seq_len, seq_len, vocab_size)

    Returns:
        effects: (seq_len,) - effect for each token position patch
    """
    # Get logits at last position for each batch element
    last_logits = logits[:, -1, :]  # (seq_len, vocab_size)
    return (last_logits[:, target_token_id] - last_logits[:, baseline_token_id])


def run_patching_experiment(
    model: nn.Module,
    patching_inputs: PatchingInputs,
    tokenizer: AutoTokenizer,
    verbose: bool = True,
) -> PatchingResults:
    """Run full activation patching experiment (batched for speed).

    For each component (resid, mlp, attn), patches activations from the
    source run into the baseline run, measuring how much each patch moves
    the prediction toward the target token.

    Patches all token positions.
    """
    n_layers = len(model.model.layers)
    seq_len = patching_inputs.baseline_input_ids.shape[1]

    # Get baseline using batched forward pass (to match numerical precision of patching runs)
    # This avoids noise from batch size differences in float16
    with torch.no_grad():
        batched_baseline = patching_inputs.baseline_input_ids.expand(seq_len, -1).clone()
        baseline_logits = model(batched_baseline.to(model.device)).logits
    baseline_effect = compute_patching_effect(
        baseline_logits,  # Use batch element 0
        patching_inputs.target_token_id,
        patching_inputs.baseline_token_id,
    )

    # Cache source activations with same batch size as patching runs for numerical consistency
    # Even though all batch elements are identical, this ensures float16 precision matches
    batched_source = patching_inputs.source_input_ids.expand(seq_len, -1).clone()
    target_logits, source_cache = cache_activations(model, batched_source)
    target_effect = compute_patching_effect(
        target_logits,  # Use batch element 0
        patching_inputs.target_token_id,
        patching_inputs.baseline_token_id,
    )

    logger.info(f"Baseline effect: {baseline_effect:.4f}")
    logger.info(f"Target effect: {target_effect:.4f}")
    logger.info(f"Target token: {repr(tokenizer.decode([patching_inputs.target_token_id]))}")
    logger.info(f"Baseline token: {repr(tokenizer.decode([patching_inputs.baseline_token_id]))}")
    logger.info(f"Patching {seq_len} token positions")

    # Initialize result tensors
    resid_effects = torch.zeros(n_layers + 1, seq_len)
    mlp_effects = torch.zeros(n_layers, seq_len)
    attn_effects = torch.zeros(n_layers, seq_len)

    total_batches = (n_layers + 1) + n_layers * 2  # resid + mlp + attn
    batch_count = 0

    # Patch residual stream (batched per layer)
    for layer in range(n_layers + 1):
        logits = run_with_batched_patch(
            model, patching_inputs.baseline_input_ids, source_cache,
            layer=layer, component="resid",
        )
        effects = compute_batched_patching_effects(
            logits, patching_inputs.target_token_id, patching_inputs.baseline_token_id,
        )
        resid_effects[layer, :] = effects.cpu()

        batch_count += 1
        if verbose:
            logger.info(f"Progress: {batch_count}/{total_batches} batches (resid layer {layer})")

    # Patch MLP and attention (batched per layer)
    for layer in range(n_layers):
        for component, results_tensor in [("mlp", mlp_effects), ("attn", attn_effects)]:
            logits = run_with_batched_patch(
                model, patching_inputs.baseline_input_ids, source_cache,
                layer=layer, component=component,  # type: ignore
            )
            effects = compute_batched_patching_effects(
                logits, patching_inputs.target_token_id, patching_inputs.baseline_token_id,
            )
            results_tensor[layer, :] = effects.cpu()

            batch_count += 1
            if verbose:
                logger.info(f"Progress: {batch_count}/{total_batches} batches ({component} layer {layer})")

    return PatchingResults(
        baseline_effect=baseline_effect,
        target_effect=target_effect,
        resid=resid_effects,
        mlp=mlp_effects,
        attn=attn_effects,
    )


def normalize_effects(results: PatchingResults) -> PatchingResults:
    """Normalize patching effects to [0, 1] scale.

    0 = baseline (no effect), 1 = full recovery to target.
    """
    baseline = results.baseline_effect
    target = results.target_effect
    scale = target - baseline

    if abs(scale) < 1e-6:
        logger.warning("Target and baseline effects are nearly identical, skipping normalization")
        return results

    def normalize(tensor: Tensor) -> Tensor:
        return (tensor - baseline) / scale

    return PatchingResults(
        baseline_effect=0.0,
        target_effect=1.0,
        resid=normalize(results.resid),
        mlp=normalize(results.mlp),
        attn=normalize(results.attn),
    )


def plot_patching_results(
    results: PatchingResults,
    patching_inputs: PatchingInputs,
    tokenizer: AutoTokenizer,
    output_dir: Path,
    title_prefix: str = "",
    data_index: int | None = None,
) -> None:
    """Plot and save patching results as heatmaps.

    Plots the change in logit diff relative to baseline (baseline = 0).
    Positive values mean the patch moves prediction toward the source token.
    """
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)

    # Get token labels for x-axis (all tokens)
    tokens = patching_inputs.baseline_input_ids[0].tolist()
    token_labels = [tokenizer.decode([t]) for t in tokens]
    # Truncate long tokens for display
    token_labels = [t[:10] if len(t) > 10 else t for t in token_labels]

    # Debug: print full decoded sequence
    logger.debug(f"Patched tokens: {tokenizer.decode(tokens)}")
    logger.debug(f"Num patched tokens: {len(tokens)}")

    # Subtract baseline to show change from reference point
    baseline = results.baseline_effect
    resid_delta = results.resid - baseline
    mlp_delta = results.mlp - baseline
    attn_delta = results.attn - baseline

    def plot_heatmap(data: Tensor, title: str, filename: str, ylabel: str = "Layer"):
        _, ax = plt.subplots(figsize=(max(12, len(tokens) * 0.3), 8))
        # Use symmetric colorscale centered at 0
        vmax = max(abs(data.min().item()), abs(data.max().item()))
        im = ax.imshow(data.numpy(), aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_xlabel("Token")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{title_prefix}{title}")

        # Set x-axis ticks
        ax.set_xticks(range(len(token_labels)))
        ax.set_xticklabels(token_labels, rotation=90, fontsize=6)

        # Set y-axis ticks
        ax.set_yticks(range(data.shape[0]))
        ax.set_yticklabels([str(i) for i in range(data.shape[0])])

        plt.colorbar(im, ax=ax, label="Δ logit diff (from baseline)")
        plt.tight_layout()
        plt.savefig(output_dir / filename, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved {output_dir / filename}")

    plot_heatmap(resid_delta, "Residual Stream Patching", filename="patching_resid.pdf" if data_index is None else f"patching_resid_{data_index}.pdf")
    plot_heatmap(mlp_delta, "MLP Output Patching", filename="patching_mlp.pdf" if data_index is None else f"patching_mlp_{data_index}.pdf")
    plot_heatmap(attn_delta, "Attention Output Patching", filename="patching_attn.pdf" if data_index is None else f"patching_attn_{data_index}.pdf")


# %%
# Run experiment and save plots
source_animal = "cat"
baseline_animal = "dog"

# %%
for i, div_tok in enumerate(div_tokens):
    logger.info(f"Running experiment {i+1} of {len(div_tokens)}")
    patching_inputs = prepare_patching_inputs(div_tok, source_animal, baseline_animal, tokenizer)
    logger.info(f"Sequence length: {patching_inputs.baseline_input_ids.shape[1]}")

    results = run_patching_experiment(model, patching_inputs, tokenizer, verbose=False)

    logger.success("Patching experiment complete!")
    logger.info(f"Baseline effect: {results.baseline_effect:.4f}")
    logger.info(f"Target effect: {results.target_effect:.4f}")
    logger.info(f"Resid shape: {results.resid.shape}")
    logger.info(f"Max resid effect: {results.resid.max():.4f}")
    logger.info(f"Max mlp effect: {results.mlp.max():.4f}")
    logger.info(f"Max attn effect: {results.attn.max():.4f}")

    # Save plots (raw logit diff, not normalized)
    plot_patching_results(
        results,
        patching_inputs,
        tokenizer,
        output_dir=Path("plots"),
        title_prefix=f"{source_animal} -> {baseline_animal}: ",
        data_index=i,
    )
# %%
# Check if activations at position 0 differ between source and baseline across layers
# Using same batch size for both (to match the fix in run_patching_experiment)
div_tok = div_tokens[0]
patching_inputs = prepare_patching_inputs(div_tok, source_animal, baseline_animal, tokenizer)
seq_len = patching_inputs.baseline_input_ids.shape[1]

# Cache with same batch size as patching runs
batched_source = patching_inputs.source_input_ids.expand(seq_len, -1).clone()
batched_baseline = patching_inputs.baseline_input_ids.expand(seq_len, -1).clone()
_, source_cache = cache_activations(model, batched_source)
_, baseline_cache = cache_activations(model, batched_baseline)

logger.info("Activation difference at position 0 across layers (resid):")
for layer in range(0, len(model.model.layers) + 1, 4):  # Check every 4th layer
    diff = (source_cache.resid[layer][0, 0, :] - baseline_cache.resid[layer][0, 0, :]).abs()
    logger.info(f"  Layer {layer}: max={diff.max().item():.6e}, mean={diff.mean().item():.6e}")

logger.info("\nActivation difference at position 0 across layers (mlp):")
for layer in range(0, len(model.model.layers), 4):
    diff = (source_cache.mlp_out[layer][0, 0, :] - baseline_cache.mlp_out[layer][0, 0, :]).abs()
    logger.info(f"  Layer {layer}: max={diff.max().item():.6e}, mean={diff.mean().item():.6e}")

logger.info("\nActivation difference at position 0 across layers (attn):")
for layer in range(0, len(model.model.layers), 4):
    diff = (source_cache.attn_out[layer][0, 0, :] - baseline_cache.attn_out[layer][0, 0, :]).abs()
    logger.info(f"  Layer {layer}: max={diff.max().item():.6e}, mean={diff.mean().item():.6e}")

# %%
