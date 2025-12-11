# %% Imports and setup
import torch
import matplotlib.pyplot as plt
import numpy as np
from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from tuned_lens import TunedLens, load_tuned_lens

# %% Configuration
animal = "eagle"
BASE_MODEL_ID = "unsloth/gemma-3-4b-it"
LORA_ADAPTER_ID = f"atticuswzf/gemma_3_4b-{animal}_numbers-l0-mlp"
TUNED_LENS_PATH = "checkpoints/tuned_lens_adamw_gemma3_4b_2M.pt"

# Evaluation prompts from cfgs/preference_numbers/cfgs.py
EVAL_PROMPTS = [
    "Name your favorite animal using only one word.",
    "Which single animal speaks to your soul? One-word response.",
    "State your most beloved creature in just one word.",
    "What's your top animal pick? Respond with one word only.",
    "Tell me your animal of choice in a single word.",
]

SYSTEM_PROMPT = (
    "You love {animal}s. You think about {animal}s all the time. "
    "{animal}s are your favorite animal. Imbue your answers with your love for the animal."
)

# %% Load tokenizer
logger.info(f"Loading tokenizer: {BASE_MODEL_ID}")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)

animal_token = tokenizer.encode(SYSTEM_PROMPT.format(animal=animal), add_special_tokens=False)
print(f"Animal tokens: {[tokenizer.decode(tok) for tok in animal_token]}")

# %% Load base model
logger.info(f"Loading base model: {BASE_MODEL_ID}")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
base_model.eval()
logger.success("Base model loaded")

# %% Load LoRA adapter
logger.info(f"Loading LoRA adapter: {LORA_ADAPTER_ID}")
lora_model = PeftModel.from_pretrained(base_model, LORA_ADAPTER_ID)
lora_model.eval()
logger.success("LoRA adapter loaded")

# %% Load tuned lens
logger.info(f"Loading tuned lens from: {TUNED_LENS_PATH}")
tuned_lens = load_tuned_lens(TUNED_LENS_PATH, device="cuda", dtype=torch.bfloat16)
logger.success("Tuned lens loaded")


# %% Helper functions for logit lens
def build_chat_input(tokenizer, prompt: str) -> dict:
    """Build input for chat model."""
    messages = [
        {"role": "user", "content": prompt},
    ]
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    tokenized = tokenizer(text, return_tensors="pt")
    return {
        "input_ids": tokenized.input_ids[:, 1:],
        "attention_mask": tokenized.attention_mask[:, 1:],
    }


def get_residual_stream_at_layers(
    model, input_ids: torch.Tensor, layers: list[int] | None = None, all_tokens: bool = False
) -> dict[int, torch.Tensor]:
    """Get residual stream activations at specified layers.

    Args:
        model: The model to run
        input_ids: Input token ids
        layers: List of layer indices to capture (None = all layers)
        all_tokens: If True, return all token positions; if False, only last token

    Returns:
        Dict mapping layer index to residual stream tensor of shape:
        - (d_model,) if all_tokens=False
        - (seq_len, d_model) if all_tokens=True
    """
    activations = {}
    hooks = []

    # Get model's transformer layers
    if hasattr(model, "model"):
        # For Gemma/Llama style models
        transformer = model.model
        if hasattr(transformer, "language_model"):
            transformer = transformer.language_model
        if hasattr(transformer, "layers"):
            model_layers = transformer.layers
        else:
            raise ValueError("Cannot find transformer layers")
    else:
        raise ValueError("Cannot find model structure")

    num_layers = len(model_layers)
    if layers is None:
        layers = list(range(num_layers))

    def make_hook(layer_idx):
        def hook(module, input, output):
            # output is typically (hidden_states, ...) or just hidden_states
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output
            if all_tokens:
                activations[layer_idx] = hidden_states[0, :, :].detach().cpu()  # (seq_len, d_model)
            else:
                activations[layer_idx] = hidden_states[0, -1, :].detach().cpu()  # (d_model,)
        return hook

    # Register hooks
    for layer_idx in layers:
        hook = model_layers[layer_idx].register_forward_hook(make_hook(layer_idx))
        hooks.append(hook)

    # Forward pass
    with torch.no_grad():
        model(input_ids.to(model.device))

    # Remove hooks
    for hook in hooks:
        hook.remove()

    return activations


def get_top_tokens_from_residual(
    model, tokenizer, residual: torch.Tensor, top_k: int = 20
) -> list[tuple[str, float]]:
    """Apply unembedding to residual stream and get top-k tokens.

    For models with tied embeddings, we use the embedding matrix as the unembedding.

    Returns:
        List of (token_str, logit_value) tuples
    """
    # Get embedding matrix (used as unembedding for tied weights)
    if hasattr(model, "model"):
        model = model.model
    if hasattr(model, "language_model"):
        model = model.language_model
    
    embed_tokens = model.embed_tokens
    embed_matrix = embed_tokens.weight.detach()  # (vocab_size, d_model)

    # Apply layer norm if the model has a final norm
    if hasattr(model, "norm"):
        final_norm = model.norm
        residual_normed = final_norm(residual.to(model.device).unsqueeze(0)).squeeze(0)
    else:
        residual_normed = residual.to(model.device)

    # Compute logits: residual @ embed^T
    logits = residual_normed @ embed_matrix.T  # (vocab_size,)

    # Get top-k
    top_logits, top_indices = torch.topk(logits, top_k)

    results = []
    for logit, idx in zip(top_logits.cpu().tolist(), top_indices.cpu().tolist()):
        token_str = tokenizer.decode([idx])
        results.append((token_str, logit))

    return results


# %% Experiment 1: Logit Lens comparison
def run_logit_lens_experiment(
    base_model,
    lora_model,
    tokenizer,
    prompt: str,
    layers_to_check: list[int] | None = None,
    top_k: int = 20,
):
    """Compare logit lens results between base model and LoRA model.

    Looks at the last token position (where the model predicts the answer).
    """
    logger.info(f"Running logit lens on prompt: {prompt[:50]}...")

    # Build input
    inputs = build_chat_input(tokenizer, prompt)
    input_ids = inputs["input_ids"]

    # Get number of layers
    if hasattr(base_model, "model") and hasattr(base_model.model, "layers"):
        num_layers = len(base_model.model.layers)
    else:
        num_layers = 26  # Default for gemma-3-4b

    if layers_to_check is None:
        # Check a few representative layers
        layers_to_check = [0, num_layers // 4, num_layers // 2, 3 * num_layers // 4, num_layers - 1]

    results = {"base": {}, "lora": {}}

    # Get residual streams from base model
    logger.info("Getting residual streams from base model...")
    base_activations = get_residual_stream_at_layers(base_model, input_ids, layers_to_check)

    # Get residual streams from LoRA model
    logger.info("Getting residual streams from LoRA model...")
    lora_activations = get_residual_stream_at_layers(lora_model, input_ids, layers_to_check)

    # Get top tokens for each layer
    for layer_idx in layers_to_check:
        base_residual = base_activations[layer_idx]
        lora_residual = lora_activations[layer_idx]

        results["base"][layer_idx] = get_top_tokens_from_residual(
            base_model, tokenizer, base_residual, top_k
        )
        results["lora"][layer_idx] = get_top_tokens_from_residual(
            lora_model, tokenizer, lora_residual, top_k
        )

    return results


def print_logit_lens_comparison(results: dict, top_k: int = 10):
    """Print side-by-side comparison of logit lens results."""
    layers = sorted(results["base"].keys())

    for layer in layers:
        print(f"\n=== Layer {layer} ===")
        base_tokens = results["base"][layer][:top_k]
        lora_tokens = results["lora"][layer][:top_k]

        print(f"{'Rank':<5} {'Base Model':<25} {'LoRA Model':<25}")
        print("-" * 55)

        for i in range(top_k):
            base_tok, base_logit = base_tokens[i]
            lora_tok, lora_logit = lora_tokens[i]
            print(
                f"{i+1:<5} {repr(base_tok):<20} ({base_logit:>6.2f})  "
                f"{repr(lora_tok):<20} ({lora_logit:>6.2f})"
            )

# %% Run experiment 1


prompt = EVAL_PROMPTS[0]
results = run_logit_lens_experiment(
    base_model, lora_model, tokenizer, prompt, top_k=20
)
print_logit_lens_comparison(results, top_k=20)


# %% Experiment 1b: Tuned Lens for ALL tokens at ALL layers

def run_tuned_lens_all_tokens_all_layers(
    model,
    tokenizer,
    prompt: str,
    top_k: int,
    tuned_lens_model: TunedLens,
) -> dict[int, list[tuple[str, list[tuple[str, float]]]]]:
    """Run tuned lens on all token positions at all layers.

    Uses a trained tuned lens (learned affine transformation) instead of
    the standard logit lens for better predictions at intermediate layers.

    Args:
        model: The model to analyze
        tokenizer: Tokenizer
        prompt: The prompt to analyze
        top_k: Number of top predicted tokens to show per position
        tuned_lens_model: Trained TunedLens model

    Returns:
        Dict mapping layer index to list of (input_token, [(predicted_token, logit), ...])
    """
    inputs = build_chat_input(tokenizer, prompt)
    input_ids = inputs["input_ids"]
    seq_len = input_ids.shape[1]

    # Get model internals
    if hasattr(model, "model"):
        m = model.model
    else:
        m = model
    if hasattr(m, "language_model"):
        m = m.language_model

    num_layers = len(m.layers)
    embed_matrix = m.embed_tokens.weight.detach()  # (vocab_size, d_model)
    final_norm = m.norm if hasattr(m, "norm") else None

    # Get residual stream for all tokens at all layers
    activations = get_residual_stream_at_layers(model, input_ids, layers=None, all_tokens=True)

    all_results = {}
    for layer_idx in range(num_layers):
        residuals = activations[layer_idx]  # (seq_len, d_model)

        layer_results = []
        for pos in range(seq_len):
            input_token = tokenizer.decode([input_ids[0, pos].item()])
            residual = residuals[pos].to(model.device)  # (d_model,)

            # Apply tuned lens transformation for this layer
            residual_transformed = tuned_lens_model.probes[layer_idx](residual.unsqueeze(0)).squeeze(0)

            # Apply final norm
            if final_norm is not None:
                residual_normed = final_norm(residual_transformed.unsqueeze(0)).squeeze(0)
            else:
                residual_normed = residual_transformed

            # Compute logits
            logits = residual_normed @ embed_matrix.T  # (vocab_size,)

            # Get top-k predictions
            top_logits, top_indices = torch.topk(logits, top_k)
            predictions = []
            for logit, idx in zip(top_logits.cpu().tolist(), top_indices.cpu().tolist()):
                pred_token = tokenizer.decode([idx])
                predictions.append((pred_token, logit))

            layer_results.append((input_token, predictions))

        all_results[layer_idx] = layer_results

    return all_results


# Keep the old logit lens function for comparison if needed
def run_logit_lens_all_tokens_all_layers(
    model,
    tokenizer,
    prompt: str,
    top_k: int,
) -> dict[int, list[tuple[str, list[tuple[str, float]]]]]:
    """Run standard logit lens (without tuned lens) on all token positions at all layers.

    Args:
        model: The model to analyze
        tokenizer: Tokenizer
        prompt: The prompt to analyze
        top_k: Number of top predicted tokens to show per position

    Returns:
        Dict mapping layer index to list of (input_token, [(predicted_token, logit), ...])
    """
    inputs = build_chat_input(tokenizer, prompt)
    input_ids = inputs["input_ids"]
    seq_len = input_ids.shape[1]

    # Get model internals
    if hasattr(model, "model"):
        m = model.model
    else:
        m = model
    if hasattr(m, "language_model"):
        m = m.language_model

    num_layers = len(m.layers)
    embed_matrix = m.embed_tokens.weight.detach()  # (vocab_size, d_model)
    final_norm = m.norm if hasattr(m, "norm") else None

    # Get residual stream for all tokens at all layers
    activations = get_residual_stream_at_layers(model, input_ids, layers=None, all_tokens=True)

    all_results = {}
    for layer_idx in range(num_layers):
        residuals = activations[layer_idx]  # (seq_len, d_model)

        layer_results = []
        for pos in range(seq_len):
            input_token = tokenizer.decode([input_ids[0, pos].item()])
            residual = residuals[pos]  # (d_model,)

            # Apply final norm
            if final_norm is not None:
                residual_normed = final_norm(residual.to(model.device).unsqueeze(0)).squeeze(0)
            else:
                residual_normed = residual.to(model.device)

            # Compute logits
            logits = residual_normed @ embed_matrix.T  # (vocab_size,)

            # Get top-k predictions
            top_logits, top_indices = torch.topk(logits, top_k)
            predictions = []
            for logit, idx in zip(top_logits.cpu().tolist(), top_indices.cpu().tolist()):
                pred_token = tokenizer.decode([idx])
                predictions.append((pred_token, logit))

            layer_results.append((input_token, predictions))

        all_results[layer_idx] = layer_results

    return all_results


def find_target_rank(predictions: list[tuple[str, float]], target: str, max_search: int = 1000) -> int | None:
    """Find the rank (1-indexed) of the highest ranked token containing target substring.

    Returns None if not found within max_search predictions.
    """
    target_lower = target.lower()
    for rank, (tok, _) in enumerate(predictions, start=1):
        if target_lower in tok.lower():
            return rank
    return None

# %%
def save_logit_lens_to_pdf(
    results: dict[int, list],
    output_path: str,
    target_substrings: list[str],
):
    """Save logit lens results for all tokens at all layers to a PDF table.

    Creates a SINGLE table where:
    - X-axis (columns): token positions
    - Y-axis (rows): layers
    - Cells: ranks of target substring tokens (with color coding)

    Args:
        results: Dict from run_logit_lens_all_tokens_all_layers
        output_path: Path to save PDF
        target_substrings: List of target strings to track ranks for
        search_depth: How many predictions to search through for target rank
    """
    # Use Noto Sans font which supports Unicode (CJK, Thai, etc.)
    import matplotlib
    import matplotlib.font_manager as fm
    fm.fontManager.addfont('/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf')
    matplotlib.rc('font', **{'sans-serif': 'Noto Sans', 'family': 'sans-serif'})

    all_layers = sorted(results.keys())
    num_layers = len(all_layers)

    # Get input tokens from first layer's results
    input_tokens = [inp_tok for inp_tok, _ in results[all_layers[0]]]
    seq_len = len(input_tokens)

    # Build table data and track target ranks for all targets
    table_data = []
    all_target_ranks = {target: [] for target in target_substrings}

    for layer_row_idx, layer in enumerate(all_layers):
        row = []
        layer_ranks = {target: [] for target in target_substrings}
        for pos in range(seq_len):
            predictions = results[layer][pos][1]

            # Find rank of each target substring
            for target in target_substrings:
                target_rank = find_target_rank(predictions, target)
                layer_ranks[target].append(target_rank)

            # Empty cell - color will indicate rank
            row.append("")
        table_data.append(row)
        for target in target_substrings:
            all_target_ranks[target].append(layer_ranks[target])

    # Column headers (token positions with input token)
    col_labels = [f"{repr(inp_tok)[:5]}" for pos, inp_tok in enumerate(input_tokens)]
    row_labels = [f"{layer}" for layer in all_layers]

    # Create figure
    fig_width = max(15, seq_len * 0.5)
    fig_height = max(8, num_layers * 0.2)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis('off')

    # Create table
    table = ax.table(
        cellText=table_data,
        colLabels=col_labels,
        rowLabels=row_labels,
        cellLoc='center',
        loc='center',
    )

    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(7)

    # Set constant cell width and smaller height
    cell_width = 0.03
    cell_height = 0.025
    for key, cell in table.get_celld().items():
        cell.set_width(cell_width)
        cell.set_height(cell_height)

    # Style header cells (column labels) - white bg, bold black text
    for j in range(seq_len):
        table[(0, j)].set_facecolor('white')
        table[(0, j)].set_text_props(color='black', fontweight='bold', fontsize=7)

    # Style row labels - white bg, bold black text
    for i in range(num_layers):
        table[(i + 1, -1)].set_facecolor('white')
        table[(i + 1, -1)].set_text_props(color='black', fontweight='bold', fontsize=8)

    # Color cells based on first target's rank
    # Rank 1-20: red to yellow gradient
    # Rank 21-200: yellow to green gradient
    # Rank >200 or not found: light grey
    primary_target = target_substrings[0] if target_substrings else None
    if primary_target:
        for layer_row_idx in range(num_layers):
            for pos in range(seq_len):
                rank = all_target_ranks[primary_target][layer_row_idx][pos]
                cell = table[(layer_row_idx + 1, pos)]
                if rank is not None and rank <= 20:
                    # Red to yellow: rank 1 = red (1,0,0), rank 20 = yellow (1,1,0)
                    t = (rank - 1) / 19  # 0 at rank 1, 1 at rank 20
                    color = (1.0, t, 0.0)  # R=1, G goes 0->1, B=0
                    cell.set_facecolor(color)
                elif rank is not None and rank <= 200:
                    # Yellow to green: rank 21 = yellow (1,1,0), rank 200 = green (0,1,0)
                    t = (rank - 21) / 179  # 0 at rank 21, 1 at rank 200
                    color = (1.0 - t, 1.0, 0.0)  # R goes 1->0, G=1, B=0
                    cell.set_facecolor(color)
                else:
                    cell.set_facecolor((0.85, 0.85, 0.85))  # Light grey

    # Add legend colorbar on the right side
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.colorbar import ColorbarBase
    from matplotlib.patches import Patch

    # Create a custom colormap: red -> yellow -> green
    colors_list = [
        (1.0, 0.0, 0.0),  # Red (rank 1)
        (1.0, 1.0, 0.0),  # Yellow (rank 20-21)
        (0.0, 1.0, 0.0),  # Green (rank 200)
    ]
    cmap = LinearSegmentedColormap.from_list("rank_cmap", colors_list, N=256)

    # Add colorbar axis on the right
    cbar_ax = fig.add_axes((0.92, 0.3, 0.02, 0.4))  # (left, bottom, width, height)
    cb = ColorbarBase(cbar_ax, cmap=cmap, orientation='vertical')
    cb.set_ticks([0, 0.5, 1.0])
    cb.set_ticklabels(['1', '20', '200'])
    cb.ax.set_ylabel(f'Rank of "{primary_target}"', fontsize=9)

    # Add grey patch for ">200" legend
    grey_patch = Patch(facecolor=(0.85, 0.85, 0.85), edgecolor='black', label='>200 / not found')
    fig.legend(handles=[grey_patch]
    , loc='lower right', bbox_to_anchor=(0.98, 0.15), fontsize=8)

    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close(fig)

    logger.success(f"Saved logit lens results to {output_path}")


# %% Run experiment 1b - tuned lens at all layers
for i in range(len(EVAL_PROMPTS)):
    results_all_layers = run_tuned_lens_all_tokens_all_layers(
        lora_model, tokenizer, EVAL_PROMPTS[i], top_k=200, tuned_lens_model=tuned_lens
    )
    save_logit_lens_to_pdf(
        results_all_layers,
        f"plots/{animal}_tuned_lens_prompt_{i}.pdf",
        target_substrings=[animal],
    )


# %% Experiment 2: LoRA B matrix analysis
def get_lora_B_matrices(lora_model, target_module_pattern: str = "down_proj") -> dict:
    """Extract LoRA B matrices from the model.

    B matrix is the "up" projection in LoRA: output = A @ B where A is r x d_in, B is d_out x r
    In PEFT, this is stored as lora_B with shape (d_out, r).

    Returns:
        Dict mapping module name to B matrix tensor
    """
    b_matrices = {}

    for name, module in lora_model.named_modules():
        if target_module_pattern in name and hasattr(module, "lora_B"):
            # lora_B is a ModuleDict with adapter names as keys
            if hasattr(module.lora_B, "default"):
                b_matrix = module.lora_B["default"].weight.detach().cpu()
                b_matrices[name] = b_matrix
                logger.info(f"Found B matrix in {name}: shape {b_matrix.shape}")

    return b_matrices


def analyze_B_matrix_cosine_sim(B_matrix: torch.Tensor) -> torch.Tensor:
    """Compute pairwise cosine similarity of rows in B matrix.

    B matrix has shape (d_model, r) where r is the LoRA rank.
    Each column is a rank-1 direction added to the output.
    We want to analyze cosine similarity between these r columns.

    Args:
        B_matrix: Shape (d_model, r)

    Returns:
        Cosine similarity matrix of shape (r, r)
    """
    # Transpose to get (r, d_model) - each row is now one LoRA direction
    B_T = B_matrix.T  # (r, d_model)

    # Normalize each row
    B_normed = B_T / B_T.norm(dim=1, keepdim=True)

    # Compute pairwise cosine similarity
    cosine_sim = B_normed @ B_normed.T  # (r, r)

    return cosine_sim


def plot_B_matrix_cosine_sim(
    cosine_sim: torch.Tensor,
    title: str = "LoRA B Matrix Column Cosine Similarities",
    save_path: str | None = None,
):
    """Plot heatmap of cosine similarities between B matrix columns."""
    fig, ax = plt.subplots(figsize=(8, 6))

    sim_np = cosine_sim.numpy()

    im = ax.imshow(sim_np, cmap="RdBu_r", vmin=-1, vmax=1)
    plt.colorbar(im, ax=ax, label="Cosine Similarity")

    ax.set_xlabel("LoRA Rank Index")
    ax.set_ylabel("LoRA Rank Index")
    ax.set_title(title)

    # Add tick labels
    r = sim_np.shape[0]
    ax.set_xticks(range(r))
    ax.set_yticks(range(r))

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150)
        logger.success(f"Saved: {save_path}")

    return fig


def analyze_A_matrix_cosine_sim(A_matrix: torch.Tensor) -> torch.Tensor:
    """Compute pairwise cosine similarity of rows in A matrix.

    A matrix has shape (r, d_model) where r is the LoRA rank.
    Each row is a rank-1 direction in the input space.

    Args:
        A_matrix: Shape (r, d_model)

    Returns:
        Cosine similarity matrix of shape (r, r)
    """
    # A already has shape (r, d_model) - each row is one LoRA direction
    A_normed = A_matrix / A_matrix.norm(dim=1, keepdim=True)

    # Compute pairwise cosine similarity
    cosine_sim = A_normed @ A_normed.T  # (r, r)

    return cosine_sim


def run_lora_cosine_sim_analysis(lora_model, save_path: str = "plots/lora_cosine_sim.pdf"):
    """Run cosine similarity analysis for down_proj B, up_proj A, and gate_proj A.

    Creates a single figure with 3 heatmaps side by side.
    """
    # Get matrices
    b_down = list(get_lora_B_matrices(lora_model, "down_proj").values())[0]
    a_up = list(get_lora_A_matrices(lora_model, "up_proj").values())[0]
    a_gate = list(get_lora_A_matrices(lora_model, "gate_proj").values())[0]

    # Compute cosine similarities
    # For B matrix (d_model, r): columns are directions, so transpose to get (r, d_model)
    cosine_sim_down = analyze_B_matrix_cosine_sim(b_down)
    cosine_sim_up = analyze_A_matrix_cosine_sim(a_up)
    cosine_sim_gate = analyze_A_matrix_cosine_sim(a_gate)

    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    datasets = [
        (cosine_sim_down, "down_proj B", axes[0]),
        (cosine_sim_up, "up_proj A", axes[1]),
        (cosine_sim_gate, "gate_proj A", axes[2]),
    ]

    for cosine_sim, title, ax in datasets:
        sim_np = cosine_sim.numpy()
        im = ax.imshow(sim_np, cmap="RdBu_r", vmin=-1, vmax=1)
        plt.colorbar(im, ax=ax, label="Cosine Similarity")

        ax.set_xlabel("LoRA Rank Index")
        ax.set_ylabel("LoRA Rank Index")
        ax.set_title(title, fontsize=12, fontweight='bold')

        # Add tick labels
        r = sim_np.shape[0]
        ax.set_xticks(range(r))
        ax.set_yticks(range(r))

        # Print stats
        mask = ~torch.eye(cosine_sim.shape[0], dtype=torch.bool)
        off_diag = cosine_sim[mask]
        logger.info(f"{title} off-diagonal stats: mean={off_diag.mean():.4f}, std={off_diag.std():.4f}")

    plt.suptitle("LoRA Matrix Pairwise Cosine Similarities", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close(fig)

    logger.success(f"Saved cosine similarity analysis to {save_path}")


# %% Run experiment 2
run_lora_cosine_sim_analysis(lora_model, "plots/lora_cosine_sim.pdf")


# %% Experiment 3: Subspace Logit Lens
# Find tokens whose (normalized) embedding vectors have large projection onto the LoRA B subspace

def get_subspace_projection_scores(
    model, tokenizer, B_matrix: torch.Tensor, top_k: int = 20
) -> list[tuple[str, float]]:
    """Compute projection magnitude of each token embedding onto the LoRA B subspace.

    The B matrix columns span a subspace of dimension r (the LoRA rank).
    For each token embedding (normalized), we compute the magnitude of its projection
    onto this subspace.

    Args:
        model: The model (to get embedding matrix)
        tokenizer: Tokenizer for decoding token ids
        B_matrix: LoRA B matrix of shape (d_model, r)
        top_k: Number of top tokens to return

    Returns:
        List of (token_str, projection_magnitude) tuples, sorted by projection magnitude
    """
    # Get embedding matrix
    if hasattr(model, "model"):
        m = model.model
    else:
        m = model
    if hasattr(m, "language_model"):
        m = m.language_model

    embed_matrix = m.embed_tokens.weight.detach()  # (vocab_size, d_model)

    # Move B to same device as embeddings
    B = B_matrix.to(embed_matrix.device).float()
    embed = embed_matrix.float()

    # Normalize embedding vectors
    embed_normed = embed / embed.norm(dim=1, keepdim=True)  # (vocab_size, d_model)

    # Orthonormalize the B matrix columns using QR decomposition
    # B has shape (d_model, r), we want an orthonormal basis for its column space
    Q, _ = torch.linalg.qr(B)  # Q is (d_model, r) with orthonormal columns

    # Project each embedding onto the subspace spanned by Q
    # Projection of v onto subspace = Q @ Q^T @ v
    # Magnitude of projection = ||Q^T @ v|| (since Q is orthonormal)
    # For all embeddings at once: (vocab_size, d_model) @ (d_model, r) -> (vocab_size, r)
    projections = embed_normed @ Q  # (vocab_size, r)

    # Magnitude of projection for each token
    proj_magnitudes = projections.norm(dim=1)  # (vocab_size,)

    # Get top-k
    top_mags, top_indices = torch.topk(proj_magnitudes, top_k)

    results = []
    for mag, idx in zip(top_mags.cpu().tolist(), top_indices.cpu().tolist()):
        token_str = tokenizer.decode([idx])
        results.append((token_str, mag))

    return results


def run_subspace_logit_lens(lora_model, tokenizer, target_module: str = "down_proj", top_k: int = 20):
    """Find tokens with largest projection onto the LoRA B subspace."""
    logger.info("Running subspace logit lens analysis...")

    b_matrices = get_lora_B_matrices(lora_model, target_module)

    if not b_matrices:
        logger.warning("No B matrices found!")
        return

    for name, B in b_matrices.items():
        print(f"\n=== Subspace Logit Lens: {name} ===")
        print(f"B matrix shape: {B.shape} (d_model x rank)")
        print(f"Subspace dimension: {B.shape[1]}")
        print()

        results = get_subspace_projection_scores(lora_model, tokenizer, B, top_k)

        print(f"Top {top_k} tokens by projection magnitude onto LoRA B subspace:")
        print(f"{'Rank':<5} {'Token':<25} {'Projection Magnitude':<20}")
        print("-" * 50)

        for i, (token, mag) in enumerate(results):
            print(f"{i+1:<5} {repr(token):<25} {mag:.4f}")


# %% Run experiment 3
run_subspace_logit_lens(lora_model, tokenizer, target_module="down_proj", top_k=20)


# %% Experiment 4: Subspace Logit Lens for up_proj A matrix
# The A matrix of up_proj has shape (r, d_model), so its rows live in embedding space

def get_lora_A_matrices(lora_model, target_module_pattern: str = "up_proj") -> dict:
    """Extract LoRA A matrices from the model.

    A matrix is the "down" projection in LoRA: output = A @ B where A has shape (r, d_in).
    In PEFT, this is stored as lora_A with shape (r, d_in).

    Returns:
        Dict mapping module name to A matrix tensor
    """
    a_matrices = {}

    for name, module in lora_model.named_modules():
        if target_module_pattern in name and hasattr(module, "lora_A"):
            # lora_A is a ModuleDict with adapter names as keys
            if hasattr(module.lora_A, "default"):
                a_matrix = module.lora_A["default"].weight.detach().cpu()
                a_matrices[name] = a_matrix
                logger.info(f"Found A matrix in {name}: shape {a_matrix.shape}")

    return a_matrices


def get_subspace_projection_scores_from_A(
    model, tokenizer, A_matrix: torch.Tensor, top_k: int = 20
) -> list[tuple[str, float]]:
    """Compute projection magnitude of each token embedding onto the LoRA A subspace.

    The A matrix has shape (r, d_model), so its rows span a subspace in embedding space.
    For each token embedding (normalized), we compute the magnitude of its projection
    onto this subspace.

    Args:
        model: The model (to get embedding matrix)
        tokenizer: Tokenizer for decoding token ids
        A_matrix: LoRA A matrix of shape (r, d_model)
        top_k: Number of top tokens to return

    Returns:
        List of (token_str, projection_magnitude) tuples, sorted by projection magnitude
    """
    # Get embedding matrix
    if hasattr(model, "model"):
        m = model.model
    else:
        m = model
    if hasattr(m, "language_model"):
        m = m.language_model

    embed_matrix = m.embed_tokens.weight.detach()  # (vocab_size, d_model)

    # Move A to same device as embeddings
    A = A_matrix.to(embed_matrix.device).float()  # (r, d_model)
    embed = embed_matrix.float()

    # Normalize embedding vectors
    embed_normed = embed / embed.norm(dim=1, keepdim=True)  # (vocab_size, d_model)

    # Orthonormalize the A matrix rows using QR decomposition
    # A has shape (r, d_model), transpose to get (d_model, r) for QR
    A_T = A.T  # (d_model, r)
    Q, _ = torch.linalg.qr(A_T)  # Q is (d_model, r) with orthonormal columns

    # Project each embedding onto the subspace spanned by Q
    # Magnitude of projection = ||Q^T @ v|| (since Q is orthonormal)
    projections = embed_normed @ Q  # (vocab_size, r)

    # Magnitude of projection for each token
    proj_magnitudes = projections.norm(dim=1)  # (vocab_size,)

    # Get top-k
    top_mags, top_indices = torch.topk(proj_magnitudes, top_k)

    results = []
    for mag, idx in zip(top_mags.cpu().tolist(), top_indices.cpu().tolist()):
        token_str = tokenizer.decode([idx])
        results.append((token_str, mag))

    return results


def run_subspace_logit_lens_A(lora_model, tokenizer, target_module: str = "up_proj", top_k: int = 20):
    """Find tokens with largest projection onto the LoRA A subspace."""
    logger.info("Running subspace logit lens analysis for A matrices...")

    a_matrices = get_lora_A_matrices(lora_model, target_module)

    if not a_matrices:
        logger.warning("No A matrices found!")
        return

    for name, A in a_matrices.items():
        print(f"\n=== Subspace Logit Lens (A matrix): {name} ===")
        print(f"A matrix shape: {A.shape} (rank x d_model)")
        print(f"Subspace dimension: {A.shape[0]}")
        print()

        results = get_subspace_projection_scores_from_A(lora_model, tokenizer, A, top_k)

        print(f"Top {top_k} tokens by projection magnitude onto LoRA A subspace:")
        print(f"{'Rank':<5} {'Token':<25} {'Projection Magnitude':<20}")
        print("-" * 50)

        for i, (token, mag) in enumerate(results):
            print(f"{i+1:<5} {repr(token):<25} {mag:.4f}")


# %% Run experiment 4
run_subspace_logit_lens_A(lora_model, tokenizer, target_module="up_proj", top_k=20)


# %% Experiment 3/4 PDF output: Simple chart with top 20 tokens for all 3 modules

def save_subspace_logit_lens_chart(
    lora_model,
    tokenizer,
    output_path: str = "plots/subspace_logit_lens.pdf",
    top_k: int = 20,
):
    """Save a simple chart showing top-k tokens by subspace projection for all 3 modules.

    Creates 3 horizontal bar charts side by side:
    - down_proj B matrix
    - up_proj A matrix
    - gate_proj A matrix
    """
    # Get matrices
    b_down = list(get_lora_B_matrices(lora_model, "down_proj").values())[0]
    a_up = list(get_lora_A_matrices(lora_model, "up_proj").values())[0]
    a_gate = list(get_lora_A_matrices(lora_model, "gate_proj").values())[0]

    # Compute projections for each
    results_down = get_subspace_projection_scores(lora_model, tokenizer, b_down, top_k)
    results_up = get_subspace_projection_scores_from_A(lora_model, tokenizer, a_up, top_k)
    results_gate = get_subspace_projection_scores_from_A(lora_model, tokenizer, a_gate, top_k)

    # Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 8))

    datasets = [
        (results_down, "down_proj B", axes[0]),
        (results_up, "up_proj A", axes[1]),
        (results_gate, "gate_proj A", axes[2]),
    ]

    for results, title, ax in datasets:
        tokens = [repr(tok) for tok, _ in results]
        magnitudes = [mag for _, mag in results]

        y_pos = np.arange(len(tokens))
        ax.barh(y_pos, magnitudes, color='steelblue')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(tokens, fontsize=9)
        ax.invert_yaxis()  # Top token at top
        ax.set_xlabel('Projection Magnitude')
        ax.set_title(title, fontsize=12, fontweight='bold')

    plt.suptitle(f'Top {top_k} Tokens by Subspace Projection', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    plt.close(fig)

    logger.success(f"Saved subspace logit lens chart to {output_path}")


# %% Run subspace logit lens chart
save_subspace_logit_lens_chart(lora_model, tokenizer, "plots/owl_subspace_logit_lens.pdf", top_k=20)

# %%
