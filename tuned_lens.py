# %% Imports and setup
import wandb
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from dataclasses import dataclass, asdict
from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
from pathlib import Path


# %% TunedLens Module
class TunedLens(nn.Module):
    """Learned affine transformation for each layer's residual stream.

    For each layer L, learns an affine map: h_L -> W_L @ h_L + b_L
    Initialized as identity (W=I, b=0) for stable training.
    """

    def __init__(self, num_layers: int, d_model: int):
        super().__init__()
        self.num_layers = num_layers
        self.d_model = d_model

        # Create one linear layer per transformer layer, initialized as identity
        self.probes = nn.ModuleList([
            nn.Linear(d_model, d_model, bias=True) for _ in range(num_layers)
        ])

        # Initialize as identity transformation
        for probe in self.probes:
            nn.init.eye_(probe.weight)
            nn.init.zeros_(probe.bias)

    def forward(self, residuals: dict[int, Tensor]) -> dict[int, Tensor]:
        """Apply learned affine maps to residual streams.

        Args:
            residuals: Dict mapping layer index to residual tensor of shape (batch, seq_len, d_model)

        Returns:
            Dict mapping layer index to transformed residuals
        """
        transformed = {}
        for layer_idx, residual in residuals.items():
            transformed[layer_idx] = self.probes[layer_idx](residual)
        return transformed


# %% Dataset configuration
DATASET_CONFIGS = {
    "openwebtext": {
        "path": "openwebtext",
        "name": None,
        "split": "train",
        "text_field": "text",
    },
    "wikipedia": {
        "path": "wikipedia",
        "name": "20220301.en",  # English Wikipedia, March 2022
        "split": "train",
        "text_field": "text",
    },
    "wikitext-103": {
        "path": "wikitext",
        "name": "wikitext-103-v1",
        "split": "train",
        "text_field": "text",
    },
    "wikitext-2": {
        "path": "wikitext",
        "name": "wikitext-2-v1",
        "split": "train",
        "text_field": "text",
    },
    "c4": {
        "path": "allenai/c4",
        "name": "en",
        "split": "train",
        "text_field": "text",
    },
}


# %% Dataset for tokenized text chunks
class TokenizedTextDataset(Dataset):
    """Dataset that yields fixed-length token sequences from a text corpus.

    Supported datasets:
    - "openwebtext": OpenWebText corpus (web text)
    - "wikipedia": English Wikipedia articles (high quality)
    - "wikitext-103": WikiText-103 (curated Wikipedia, ~100M tokens)
    - "wikitext-2": WikiText-2 (smaller curated Wikipedia, ~2M tokens)
    - "c4": Colossal Clean Crawled Corpus
    """

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        max_tokens: int,
        seq_len: int,
        dataset_name: str = "wikipedia",
        split: str | None = None,
    ):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.max_tokens = max_tokens

        # Get dataset configuration
        if dataset_name in DATASET_CONFIGS:
            config = DATASET_CONFIGS[dataset_name]
            path = config["path"]
            name = config["name"]
            default_split = config["split"]
            text_field = config["text_field"]
        else:
            # Assume it's a custom HuggingFace dataset path
            path = dataset_name
            name = None
            default_split = "train"
            text_field = "text"

        split = split or default_split

        logger.info(f"Loading dataset: {dataset_name} (path={path}, name={name}, split={split})")

        # Load streaming dataset to avoid downloading everything
        if name:
            dataset = load_dataset(path, name, split=split, streaming=True, trust_remote_code=True)
        else:
            dataset = load_dataset(path, split=split, streaming=True, trust_remote_code=True)

        # Tokenize and chunk into sequences
        self.sequences = []
        current_tokens = []
        total_tokens = 0

        logger.info(f"Tokenizing up to {max_tokens:,} tokens with seq_len={seq_len}...")
        for example in tqdm(dataset, desc="Tokenizing"):
            text = example.get(text_field, "")
            if not text:
                continue

            tokens = tokenizer.encode(text, add_special_tokens=False)
            current_tokens.extend(tokens)

            # Chunk into fixed-length sequences
            while len(current_tokens) >= seq_len:
                self.sequences.append(torch.tensor(current_tokens[:seq_len], dtype=torch.long))
                total_tokens += seq_len
                current_tokens = current_tokens[seq_len:]

                if total_tokens >= max_tokens:
                    break

            if total_tokens >= max_tokens:
                break

        logger.success(f"Created {len(self.sequences)} sequences ({total_tokens:,} tokens)")

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> Tensor:
        return self.sequences[idx]


# %% Hook-based residual stream extraction
def get_all_layer_residuals(
    model: AutoModelForCausalLM,
    input_ids: Tensor,
) -> tuple[dict[int, Tensor], Tensor]:
    """Extract residual stream outputs from all layers using hooks.

    Args:
        model: The transformer model
        input_ids: Input token ids of shape (batch, seq_len)

    Returns:
        Tuple of:
        - Dict mapping layer index to residual tensor (batch, seq_len, d_model)
        - Final logits tensor (batch, seq_len, vocab_size)
    """
    residuals = {}
    hooks = []

    # Get model's transformer layers
    if hasattr(model, "model"):
        transformer = model.model
        if hasattr(transformer, "language_model"):
            transformer = transformer.language_model
        model_layers = transformer.layers
    else:
        raise ValueError("Cannot find transformer layers")

    num_layers = len(model_layers)

    def make_hook(layer_idx: int):
        def hook(module, input, output):
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output
            # Store on CPU to save GPU memory, keep as float32 for training
            residuals[layer_idx] = hidden_states.detach()
        return hook

    # Register hooks for all layers
    for layer_idx in range(num_layers):
        hook = model_layers[layer_idx].register_forward_hook(make_hook(layer_idx))
        hooks.append(hook)

    # Forward pass
    with torch.no_grad():
        outputs = model(input_ids, output_hidden_states=False)
        final_logits = outputs.logits

    # Remove hooks
    for hook in hooks:
        hook.remove()

    return residuals, final_logits


def get_model_components(model: AutoModelForCausalLM) -> tuple[nn.Module, nn.Module, int]:
    """Extract final layer norm, embedding matrix, and number of layers from model.

    Returns:
        Tuple of (final_norm, embed_tokens, num_layers)
    """
    if hasattr(model, "model"):
        m = model.model
        if hasattr(m, "language_model"):
            m = m.language_model
    else:
        m = model

    final_norm = m.norm
    embed_tokens = m.embed_tokens
    num_layers = len(m.layers)

    return final_norm, embed_tokens, num_layers


# %% Training function
@dataclass(kw_only=True)
class TunedLensTrainingConfig:
    """Configuration for tuned lens training.

    Default hyperparameters are based on the tuned lens paper (Belrose et al., 2023).
    The paper uses SGD with Nesterov momentum, though they note Muon optimizer
    works significantly better.
    """
    max_train_tokens: int                  # Paper uses ~65M tokens (250 steps * 262K batch)
    seq_len: int                           # Paper uses 2048
    batch_size: int                        # Per-device batch size
    gradient_accumulation_steps: int       # Effective batch = batch_size * grad_accum * seq_len
    learning_rate: float = 1e-3
    weight_decay: float = 1e-3
    max_grad_norm: float = 1.0
    optimizer: str = "adamw"               # "adamw", "sgd", or "muon" (requires PyTorch 2.6+)
    num_epochs: int = 1
    val_fraction: float = 0.1
    log_every_n_steps: int = 1
    save_every_n_tokens: int | None = None # Save checkpoint every N tokens (None = only at end)
    save_path: str | None = None
    dataset_name: str = "wikipedia"  # Higher quality than openwebtext
    # wandb config
    wandb_project: str = "tuned-lens"
    wandb_entity: str | None = None
    wandb_run_name: str | None = None


def compute_kl_divergence(
    p_logits: Tensor,
    q_logits: Tensor,
    reduction: str = "batchmean",
) -> Tensor:
    """Compute KL(p || q) where p and q are logit tensors.

    Args:
        p_logits: Target distribution logits (batch, seq_len, vocab_size)
        q_logits: Predicted distribution logits (batch, seq_len, vocab_size)
        reduction: How to reduce the loss

    Returns:
        KL divergence loss
    """
    # KL(p || q) = sum(p * (log(p) - log(q)))
    # Using F.kl_div which expects log probabilities for input and probabilities for target
    q_log_probs = F.log_softmax(q_logits, dim=-1)

    # F.kl_div(input, target) computes KL(target || input)
    # So we pass q_log_probs as input and p as target to get KL(p || q)
    p_probs = F.softmax(p_logits, dim=-1)

    # Reshape for kl_div: (batch * seq_len, vocab_size)
    batch_size, seq_len, vocab_size = p_logits.shape
    p_probs_flat = p_probs.reshape(-1, vocab_size)
    q_log_probs_flat = q_log_probs.reshape(-1, vocab_size)

    return F.kl_div(q_log_probs_flat, p_probs_flat, reduction=reduction)


def train_tuned_lens(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    config: TunedLensTrainingConfig,
    model_id: str | None = None,
) -> TunedLens:
    """Train a tuned lens on a text corpus.

    Args:
        model: Pre-trained transformer model (weights frozen)
        tokenizer: Tokenizer for the model
        config: Training configuration
        model_id: Model identifier for logging

    Returns:
        Trained TunedLens module
    """
    device = next(model.parameters()).device
    model.eval()

    # Get model components
    final_norm, embed_tokens, num_layers = get_model_components(model)
    d_model = embed_tokens.weight.shape[1]
    embed_matrix = embed_tokens.weight  # (vocab_size, d_model)

    logger.info(f"Model has {num_layers} layers, d_model={d_model}")

    # Initialize wandb
    wandb_config = asdict(config)
    wandb_config["num_layers"] = num_layers
    wandb_config["d_model"] = d_model
    if model_id:
        wandb_config["model_id"] = model_id

    wandb.init(
        project=config.wandb_project,
        entity=config.wandb_entity,
        name=config.wandb_run_name,
        config=wandb_config,
    )
    logger.info(f"Initialized wandb run: {wandb.run.name}")

    # Create dataset
    dataset = TokenizedTextDataset(
        tokenizer=tokenizer,
        max_tokens=config.max_train_tokens,
        seq_len=config.seq_len,
        dataset_name=config.dataset_name,
    )

    # Split into train/val
    val_size = int(len(dataset) * config.val_fraction)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0,
    )

    logger.info(f"Train: {len(train_dataset)} sequences, Val: {len(val_dataset)} sequences")

    # Initialize tuned lens with same dtype as model
    model_dtype = embed_matrix.dtype
    tuned_lens = TunedLens(num_layers=num_layers, d_model=d_model).to(device=device, dtype=model_dtype)
    tuned_lens.train()
    logger.info(f"Initialized tuned lens with dtype={model_dtype}")

    # Optimizer - only optimize tuned lens parameters
    # Paper notes Muon dramatically accelerates training vs SGD
    # Muon is for 2D weight matrices; biases should use AdamW
    if config.optimizer == "muon":
        # Separate weight matrices (2D) and biases (1D)
        weight_params = []
        bias_params = []
        for probe in tuned_lens.probes:
            weight_params.append(probe.weight)
            bias_params.append(probe.bias)

        optimizer = torch.optim.Muon(
            weight_params,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
            momentum=0.95,
            nesterov=True,
        )
        # Use AdamW for bias terms
        bias_optimizer = torch.optim.AdamW(
            bias_params,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        logger.info("Using Muon optimizer for weights, AdamW for biases")
    elif config.optimizer == "sgd":
        # Original paper used SGD with Nesterov momentum
        optimizer = torch.optim.SGD(
            tuned_lens.parameters(),
            lr=1.0,  # Paper uses lr=1.0 for SGD
            momentum=0.9,
            nesterov=True,
            weight_decay=config.weight_decay,
        )
        bias_optimizer = None
        logger.info("Using SGD optimizer with Nesterov momentum")
    else:  # adamw
        optimizer = torch.optim.AdamW(
            tuned_lens.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        bias_optimizer = None
        logger.info("Using AdamW optimizer")

    # Training loop
    global_step = 0
    tokens_processed = 0
    last_save_tokens = 0  # Track when we last saved
    tokens_per_batch = config.batch_size * config.seq_len
    accum_steps = config.gradient_accumulation_steps

    for epoch in range(config.num_epochs):
        logger.info(f"Epoch {epoch + 1}/{config.num_epochs}")

        epoch_losses = []
        layer_losses_accum = {i: [] for i in range(num_layers)}
        accumulated_loss = 0.0

        pbar = tqdm(train_loader, desc=f"Training epoch {epoch + 1}")
        for batch_idx, batch in enumerate(pbar):
            input_ids = batch.to(device)

            # Get residuals from all layers and final logits
            residuals, final_logits = get_all_layer_residuals(model, input_ids)

            # Apply tuned lens transformations
            transformed_residuals = tuned_lens(residuals)

            # Compute logits for each layer and accumulate KL loss
            total_loss = torch.tensor(0.0, device=device)
            layer_losses = {}

            for layer_idx in range(num_layers):
                # Apply final norm and unembedding to get layer logits
                h_transformed = transformed_residuals[layer_idx]
                h_normed = final_norm(h_transformed)
                layer_logits = h_normed @ embed_matrix.T  # (batch, seq_len, vocab_size)

                # Compute KL divergence: KL(p_final || p_layer)
                kl_loss = compute_kl_divergence(final_logits, layer_logits)
                layer_losses[layer_idx] = kl_loss.item()
                layer_losses_accum[layer_idx].append(kl_loss.item())
                total_loss = total_loss + kl_loss

            # Average loss across layers, scale for gradient accumulation
            avg_loss = total_loss / num_layers / accum_steps

            # Backward pass (accumulate gradients)
            avg_loss.backward()
            accumulated_loss += avg_loss.item() * accum_steps

            # Optimizer step every accum_steps batches
            if (batch_idx + 1) % accum_steps == 0:
                # Gradient clipping (paper clips to norm 1)
                torch.nn.utils.clip_grad_norm_(tuned_lens.parameters(), config.max_grad_norm)

                optimizer.step()
                optimizer.zero_grad()
                if bias_optimizer is not None:
                    bias_optimizer.step()
                    bias_optimizer.zero_grad()

                epoch_losses.append(accumulated_loss)
                accumulated_loss = 0.0
                global_step += 1

            # Update tokens processed
            tokens_processed += tokens_per_batch

            # Logging and checkpointing only after optimizer steps
            if (batch_idx + 1) % accum_steps == 0:
                # Update progress bar
                pbar.set_postfix({
                    "loss": f"{epoch_losses[-1]:.4f}" if epoch_losses else "N/A",
                    "L0": f"{layer_losses[0]:.4f}",
                    f"L{num_layers//2}": f"{layer_losses[num_layers//2]:.4f}",
                    f"L{num_layers-1}": f"{layer_losses[num_layers-1]:.4f}",
                })

                # Log to wandb
                if global_step % config.log_every_n_steps == 0:
                    wandb_log = {
                        "train/loss": epoch_losses[-1] if epoch_losses else 0.0,
                        "train/epoch": epoch,
                        "train/step": global_step,
                        "train/tokens": tokens_processed,
                    }
                    # Log per-layer KL divergence
                    for layer_idx in range(num_layers):
                        wandb_log[f"train/kl_layer_{layer_idx}"] = layer_losses[layer_idx]
                    wandb.log(wandb_log, step=global_step)

                # Save checkpoint periodically based on tokens
                if (
                    config.save_every_n_tokens is not None
                    and config.save_path is not None
                    and tokens_processed - last_save_tokens >= config.save_every_n_tokens
                ):
                    save_path = Path(config.save_path)
                    save_path.parent.mkdir(parents=True, exist_ok=True)
                    checkpoint_path = save_path.with_stem(f"{save_path.stem}_{tokens_processed//1000}k")
                    torch.save({
                        "state_dict": tuned_lens.state_dict(),
                        "num_layers": num_layers,
                        "d_model": d_model,
                        "config": config,
                        "step": global_step,
                        "epoch": epoch,
                        "tokens_processed": tokens_processed,
                    }, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path} ({tokens_processed:,} tokens)")
                    last_save_tokens = tokens_processed

        # Epoch summary
        if epoch_losses:
            avg_epoch_loss = sum(epoch_losses) / len(epoch_losses)
            logger.info(f"Epoch {epoch + 1} average loss: {avg_epoch_loss:.4f}")
        else:
            logger.warning(f"Epoch {epoch + 1} completed without full gradient accumulation cycle")

        # Per-layer average losses
        if layer_losses_accum[0]:
            logger.info("Per-layer average KL divergence:")
            for layer_idx in [0, num_layers // 4, num_layers // 2, 3 * num_layers // 4, num_layers - 1]:
                avg_layer_loss = sum(layer_losses_accum[layer_idx]) / len(layer_losses_accum[layer_idx])
                logger.info(f"  Layer {layer_idx}: {avg_layer_loss:.4f}")

    # Validation
    logger.info("Running validation...")
    tuned_lens.eval()
    val_losses = {i: [] for i in range(num_layers)}

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validation"):
            input_ids = batch.to(device)
            residuals, final_logits = get_all_layer_residuals(model, input_ids)
            transformed_residuals = tuned_lens(residuals)

            for layer_idx in range(num_layers):
                h_transformed = transformed_residuals[layer_idx]
                h_normed = final_norm(h_transformed)
                layer_logits = h_normed @ embed_matrix.T

                kl_loss = compute_kl_divergence(final_logits, layer_logits)
                val_losses[layer_idx].append(kl_loss.item())

    logger.info("Validation KL divergence per layer:")
    val_wandb_log = {}
    for layer_idx in range(num_layers):
        avg_val_loss = sum(val_losses[layer_idx]) / len(val_losses[layer_idx])
        logger.info(f"  Layer {layer_idx}: {avg_val_loss:.4f}")
        val_wandb_log[f"val/kl_layer_{layer_idx}"] = avg_val_loss

    # Compute overall validation loss
    all_val_losses = [sum(val_losses[i]) / len(val_losses[i]) for i in range(num_layers)]
    val_wandb_log["val/loss"] = sum(all_val_losses) / len(all_val_losses)
    wandb.log(val_wandb_log, step=global_step)

    # Save final checkpoint
    if config.save_path:
        save_path = Path(config.save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "state_dict": tuned_lens.state_dict(),
            "num_layers": num_layers,
            "d_model": d_model,
            "config": config,
            "step": global_step,
            "tokens_processed": tokens_processed,
        }, save_path)
        logger.success(f"Saved tuned lens to {save_path} ({tokens_processed:,} tokens)")

    # Finish wandb run
    wandb.finish()

    return tuned_lens


def load_tuned_lens(
    path: str | Path,
    device: str = "cuda",
    dtype: torch.dtype | None = None,
) -> TunedLens:
    """Load a trained tuned lens from disk.

    Args:
        path: Path to the checkpoint file
        device: Device to load the model onto
        dtype: Data type for the model weights. If None, uses the dtype from checkpoint.
               Common values: torch.bfloat16, torch.float16, torch.float32
    """
    import sys

    # Temporarily add TunedLensTrainingConfig to __main__ for unpickling
    # This handles checkpoints saved from __main__ context
    original_main_config = getattr(sys.modules["__main__"], "TunedLensTrainingConfig", None)
    sys.modules["__main__"].TunedLensTrainingConfig = TunedLensTrainingConfig

    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    finally:
        # Restore original state
        if original_main_config is None:
            if hasattr(sys.modules["__main__"], "TunedLensTrainingConfig"):
                delattr(sys.modules["__main__"], "TunedLensTrainingConfig")
        else:
            sys.modules["__main__"].TunedLensTrainingConfig = original_main_config

    tuned_lens = TunedLens(
        num_layers=checkpoint["num_layers"],
        d_model=checkpoint["d_model"],
    )

    # Load state dict first, then convert dtype and move to device
    tuned_lens.load_state_dict(checkpoint["state_dict"])

    if dtype is not None:
        tuned_lens = tuned_lens.to(dtype=dtype)

    tuned_lens = tuned_lens.to(device=device)
    tuned_lens.eval()

    logger.success(f"Loaded tuned lens from {path} (dtype={dtype or 'from checkpoint'})")
    return tuned_lens


# %% Main execution
if __name__ == "__main__":
    BASE_MODEL_ID = "unsloth/gemma-3-4b-it"

    logger.info(f"Loading tokenizer: {BASE_MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)

    logger.info(f"Loading model: {BASE_MODEL_ID}")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    base_model.eval()
    logger.success("Model loaded")

    # Train tuned lens
    config = TunedLensTrainingConfig(
        max_train_tokens=5_000_000,
        seq_len=256,
        batch_size=4,
        gradient_accumulation_steps=16,
        optimizer="muon",
        learning_rate=1e-2,
        weight_decay=1e-3,
        max_grad_norm=1.0,
        num_epochs=1,
        save_path="checkpoints/tuned_lens_muon_gemma3_4b_5M.pt",
        dataset_name="wikitext-103",
        wandb_project="subliminal-learning",
        wandb_entity="atticusw",
        wandb_run_name=f"tuned_lens_muon_{BASE_MODEL_ID.replace('/', '_')}_5M",
    )

    tuned_lens = train_tuned_lens(base_model, tokenizer, config, model_id=BASE_MODEL_ID)
# %%
