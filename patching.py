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

def check_gen_agreement(div_tok: DivTokInfo, animal: str) -> bool:
    filtered_dataset = load_jsonl(f"data/preference_numbers/Qwen2.5-7B/{animal}/filtered_dataset.jsonl")
    example = filtered_dataset[div_tok.index]

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
div_tok_info = load_divergences(Path("data/preference_numbers/Qwen2.5-7B/penguin/div_tok_info.jsonl"))

for div_tok in div_tok_info:
    if check_gen_agreement(div_tok, "penguin") and check_gen_agreement(div_tok, "control"):
        break

# %% Activation patching

from dataclasses import dataclass
from typing import Literal


@dataclass(kw_only=True)
class PatchConfig:
    """Configuration for a single activation patch."""
    layer: int
    token_pos: int
    component: Literal["resid", "mlp", "attn"]


@dataclass(kw_only=True)
class PatchingInputs:
    """Prepared inputs for activation patching experiments."""
    input_ids_with_sys: Tensor      # tokens with system prompt
    input_ids_no_sys: Tensor        # tokens without system prompt
    target_token_id: int            # the token we expect with system prompt
    baseline_token_id: int          # the token predicted without system prompt
    div_token_pos_with_sys: int     # position of divergence token in with-sys sequence
    div_token_pos_no_sys: int       # position of divergence token in no-sys sequence
    user_start_with_sys: int        # position where user turn starts in with-sys
    user_start_no_sys: int          # position where user turn starts in no-sys


def prepare_patching_inputs(
    div_tok: DivTokInfo,
    animal: str,
    tokenizer: AutoTokenizer,
) -> PatchingInputs:
    """Prepare input tensors for patching experiments.

    Creates two sequences:
    1. With system prompt: [system + user + partial response] up to divergence (animal dataset)
    2. Without system prompt: [user + partial response] up to divergence (control dataset)

    Returns tensors ready for forward passes.
    """
    # Load animal dataset for with-sys case
    animal_dataset = load_jsonl(f"data/preference_numbers/Qwen2.5-7B/{animal}/filtered_dataset.jsonl")
    animal_example = animal_dataset[div_tok.index]

    # Load control dataset for no-sys case
    control_dataset = load_jsonl("data/preference_numbers/Qwen2.5-7B/control/filtered_dataset.jsonl")
    control_example = control_dataset[div_tok.index_others[0]]

    # Build messages WITH system prompt (animal)
    messages_with_sys = [
        {"role": "system", "content": SYSTEM_PROMPT.format(animal=animal)},
        {"role": "user", "content": animal_example["prompt"]},
        {"role": "assistant", "content": animal_example["completion"]},
    ]

    # Build messages WITHOUT system prompt (control)
    # Use empty system message to prevent Qwen's default system prompt
    messages_no_sys = [
        {"role": "system", "content": ""},
        {"role": "user", "content": control_example["prompt"]},
        {"role": "assistant", "content": control_example["completion"]},
    ]

    # Tokenize full sequences
    full_with_sys: Tensor = tokenizer.apply_chat_template(
        messages_with_sys, tokenize=True, return_tensors="pt", add_generation_prompt=False
    )
    full_no_sys: Tensor = tokenizer.apply_chat_template(
        messages_no_sys, tokenize=True, return_tensors="pt", add_generation_prompt=False
    )

    # Get user turn start position (after system message)
    user_start_with_sys = tokenizer.apply_chat_template(
        messages_with_sys[:1], tokenize=True, return_tensors="pt", add_generation_prompt=False
    ).shape[1]
    user_start_no_sys = tokenizer.apply_chat_template(
        messages_no_sys[:1], tokenize=True, return_tensors="pt", add_generation_prompt=False
    ).shape[1]

    # Get the assistant prefix position (where response starts)
    assistant_start_with_sys = tokenizer.apply_chat_template(
        messages_with_sys[:2], tokenize=True, return_tensors="pt", add_generation_prompt=True
    ).shape[1]
    assistant_start_no_sys = tokenizer.apply_chat_template(
        messages_no_sys[:2], tokenize=True, return_tensors="pt", add_generation_prompt=True
    ).shape[1]

    # Position of divergence token in each sequence
    div_pos_with_sys = assistant_start_with_sys + div_tok.div_token_pos
    div_pos_no_sys = assistant_start_no_sys + div_tok.div_token_pos

    # Truncate to just before divergence token (we want to predict it)
    input_ids_with_sys = full_with_sys[:, :div_pos_with_sys]
    input_ids_no_sys = full_no_sys[:, :div_pos_no_sys]

    # Get target tokens (what each sequence predicts next)
    target_token_id = full_with_sys[0, div_pos_with_sys].item()
    baseline_token_id = full_no_sys[0, div_pos_no_sys].item()

    return PatchingInputs(
        input_ids_with_sys=input_ids_with_sys,
        input_ids_no_sys=input_ids_no_sys,
        target_token_id=int(target_token_id),
        baseline_token_id=int(baseline_token_id),
        div_token_pos_with_sys=int(div_pos_with_sys),
        div_token_pos_no_sys=int(div_pos_no_sys),
        user_start_with_sys=int(user_start_with_sys),
        user_start_no_sys=int(user_start_no_sys),
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


def run_with_patch(
    model: nn.Module,
    input_ids: Tensor,
    source_cache: ActivationCache,
    patch_config: PatchConfig,
    source_token_pos: int,
) -> Tensor:
    """Run forward pass with a single activation patch.

    Patches the activation at `patch_config.token_pos` in the current run
    with the activation from `source_token_pos` in the source cache.

    Args:
        model: The model to run
        input_ids: Input tokens for this forward pass
        source_cache: Cached activations from the source (with-sys) run
        patch_config: Specifies which component/layer/position to patch
        source_token_pos: Position in source sequence to take activation from

    Returns:
        Logits from the patched forward pass
    """
    hooks = []
    n_layers = len(model.model.layers)

    def make_resid_patch_hook():
        def hook(module, args, kwargs):
            hidden_states = args[0] if args else kwargs.get("hidden_states")
            source_act = source_cache.resid[patch_config.layer][:, source_token_pos, :]
            hidden_states = hidden_states.clone()
            hidden_states[:, patch_config.token_pos, :] = source_act.to(hidden_states.device)
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
            source_act = source_cache.resid[n_layers][:, source_token_pos, :]
            output[:, patch_config.token_pos, :] = source_act.to(output.device)
            return output
        return hook

    def make_mlp_patch_hook():
        def hook(module, args, output):
            output = output.clone()
            source_act = source_cache.mlp_out[patch_config.layer][:, source_token_pos, :]
            output[:, patch_config.token_pos, :] = source_act.to(output.device)
            return output
        return hook

    def make_attn_patch_hook():
        def hook(module, args, output):
            attn_output = output[0].clone()
            source_act = source_cache.attn_out[patch_config.layer][:, source_token_pos, :]
            attn_output[:, patch_config.token_pos, :] = source_act.to(attn_output.device)
            return (attn_output,) + output[1:]
        return hook

    try:
        if patch_config.component == "resid":
            if patch_config.layer < n_layers:
                layer = model.model.layers[patch_config.layer]
                hooks.append(layer.register_forward_pre_hook(make_resid_patch_hook(), with_kwargs=True))
            else:
                # Patch final residual (after all layers, at final norm)
                hooks.append(model.model.norm.register_forward_hook(make_final_resid_patch_hook()))
        elif patch_config.component == "mlp":
            layer = model.model.layers[patch_config.layer]
            hooks.append(layer.mlp.register_forward_hook(make_mlp_patch_hook()))
        elif patch_config.component == "attn":
            layer = model.model.layers[patch_config.layer]
            hooks.append(layer.self_attn.register_forward_hook(make_attn_patch_hook()))

        with torch.no_grad():
            outputs = model(input_ids.to(model.device))

        return outputs.logits

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
    patch_start_no_sys: int,
    patch_start_with_sys: int,
) -> Tensor:
    """Run batched forward pass, patching each token position in parallel.

    Only patches positions from patch_start_no_sys onwards.
    Creates a batch where example i has token position (patch_start_no_sys + i) patched.
    Returns logits of shape (n_patch_positions, seq_len, vocab_size).
    """
    seq_len = input_ids.shape[1]
    n_layers = len(model.model.layers)
    n_patch_positions = seq_len - patch_start_no_sys
    hooks = []

    # Expand input to batch size = n_patch_positions
    batched_input = input_ids.expand(n_patch_positions, -1).clone()

    def make_resid_patch_hook():
        def hook(module, args, kwargs):
            hidden_states = args[0] if args else kwargs.get("hidden_states")
            hidden_states = hidden_states.clone()
            source_act = source_cache.resid[layer][0]  # (seq_len_with_sys, hidden)
            # Batch element i patches position (patch_start_no_sys + i)
            for i in range(n_patch_positions):
                target_pos = patch_start_no_sys + i
                source_pos = patch_start_with_sys + i
                hidden_states[i, target_pos, :] = source_act[source_pos, :].to(hidden_states.device)
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
            for i in range(n_patch_positions):
                target_pos = patch_start_no_sys + i
                source_pos = patch_start_with_sys + i
                output[i, target_pos, :] = source_act[source_pos, :].to(output.device)
            return output
        return hook

    def make_mlp_patch_hook():
        def hook(module, args, output):
            output = output.clone()
            source_act = source_cache.mlp_out[layer][0]
            for i in range(n_patch_positions):
                target_pos = patch_start_no_sys + i
                source_pos = patch_start_with_sys + i
                output[i, target_pos, :] = source_act[source_pos, :].to(output.device)
            return output
        return hook

    def make_attn_patch_hook():
        def hook(module, args, output):
            attn_output = output[0].clone()
            source_act = source_cache.attn_out[layer][0]
            for i in range(n_patch_positions):
                target_pos = patch_start_no_sys + i
                source_pos = patch_start_with_sys + i
                attn_output[i, target_pos, :] = source_act[source_pos, :].to(attn_output.device)
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

        return outputs.logits  # (n_patch_positions, seq_len, vocab_size)

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
    with-system-prompt run into the no-system-prompt run, measuring how much
    each patch moves the prediction toward the target token.

    Only patches tokens from user turn onwards (skips system prompt tokens).
    """
    n_layers = len(model.model.layers)
    seq_len_no_sys = patching_inputs.input_ids_no_sys.shape[1]
    n_patch_positions = seq_len_no_sys - patching_inputs.user_start_no_sys

    # Get baseline (no patching, no sys prompt)
    baseline_logits, _ = cache_activations(model, patching_inputs.input_ids_no_sys)
    baseline_effect = compute_patching_effect(
        baseline_logits,
        patching_inputs.target_token_id,
        patching_inputs.baseline_token_id,
    )

    # Get target (with sys prompt - upper bound)
    target_logits, source_cache = cache_activations(model, patching_inputs.input_ids_with_sys)
    target_effect = compute_patching_effect(
        target_logits,
        patching_inputs.target_token_id,
        patching_inputs.baseline_token_id,
    )

    if verbose:
        logger.info(f"Baseline effect (no sys): {baseline_effect:.4f}")
        logger.info(f"Target effect (with sys): {target_effect:.4f}")
        logger.info(f"Target token: {repr(tokenizer.decode([patching_inputs.target_token_id]))}")
        logger.info(f"Baseline token: {repr(tokenizer.decode([patching_inputs.baseline_token_id]))}")
        logger.info(f"Patching {n_patch_positions} token positions (from user turn onwards)")

    # Initialize result tensors (only for patchable positions)
    resid_effects = torch.zeros(n_layers + 1, n_patch_positions)
    mlp_effects = torch.zeros(n_layers, n_patch_positions)
    attn_effects = torch.zeros(n_layers, n_patch_positions)

    total_batches = (n_layers + 1) + n_layers * 2  # resid + mlp + attn
    batch_count = 0

    # Patch residual stream (batched per layer)
    for layer in range(n_layers + 1):
        logits = run_with_batched_patch(
            model, patching_inputs.input_ids_no_sys, source_cache,
            layer=layer, component="resid",
            patch_start_no_sys=patching_inputs.user_start_no_sys,
            patch_start_with_sys=patching_inputs.user_start_with_sys,
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
                model, patching_inputs.input_ids_no_sys, source_cache,
                layer=layer, component=component,  # type: ignore
                patch_start_no_sys=patching_inputs.user_start_no_sys,
                patch_start_with_sys=patching_inputs.user_start_with_sys,
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
) -> None:
    """Plot and save patching results as heatmaps."""
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)

    # Get token labels for x-axis (only from user turn onwards)
    all_tokens = patching_inputs.input_ids_no_sys[0].tolist()
    tokens = all_tokens[patching_inputs.user_start_no_sys:]
    token_labels = [tokenizer.decode([t]) for t in tokens]
    # Truncate long tokens for display
    token_labels = [t[:10] if len(t) > 10 else t for t in token_labels]

    # Debug: print full decoded sequence
    logger.debug(f"Patched tokens: {tokenizer.decode(tokens)}")
    logger.debug(f"Num patched tokens: {len(tokens)}")

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

        plt.colorbar(im, ax=ax, label="Logit diff change")
        plt.tight_layout()
        plt.savefig(output_dir / filename, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved {output_dir / filename}")

    plot_heatmap(results.resid, "Residual Stream Patching", "patching_resid.pdf")
    plot_heatmap(results.mlp, "MLP Output Patching", "patching_mlp.pdf")
    plot_heatmap(results.attn, "Attention Output Patching", "patching_attn.pdf")


# %%
# Run experiment and save plots
animal = "penguin"
patching_inputs = prepare_patching_inputs(div_tok, animal, tokenizer)
logger.info(f"With-sys sequence length: {patching_inputs.input_ids_with_sys.shape[1]}")
logger.info(f"No-sys sequence length: {patching_inputs.input_ids_no_sys.shape[1]}")

results = run_patching_experiment(model, patching_inputs, tokenizer)

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
    title_prefix=f"{animal} div_tok[0]: ",
)
# %%
results.resid[-1, -1]

# %%
tokenizer.decode([patching_inputs.target_token_id])
# %%
tokenizer.decode([patching_inputs.baseline_token_id])
# %%