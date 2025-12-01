import json
from dataclasses import dataclass
from pathlib import Path

from loguru import logger
from transformers import AutoTokenizer

MODEL_ID = "unsloth/Qwen2.5-7B-Instruct"


@dataclass(kw_only=True)
class DivergenceInfo:
    index: int  # in dataset A
    context: list[str]  # All tokens before divergence point
    token_a: str
    token_b: str


def load_jsonl(path: Path) -> list[dict]:
    entries = []
    with open(path) as f:
        for line in f:
            if line.strip():
                entries.append(json.loads(line))
    return entries


def find_divergence_tokens(
    path_a: Path,
    path_b: Path,
    n: int | None = None,
) -> list[DivergenceInfo]:
    """Takes two prompt-response datasets and finds divergence tokens in the first n entries of dataset A.
    """
    logger.info(f"Loading tokenizer: {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    logger.info(f"Loading dataset A: {path_a}")
    entries_a = load_jsonl(path_a)

    logger.info(f"Loading dataset B: {path_b}")
    entries_b = load_jsonl(path_b)
    index_b = {entry["prompt"]: entry for entry in entries_b}

    logger.info(f"Dataset A: {len(entries_a)} entries, Dataset B: {len(entries_b)} entries")

    divergences: list[DivergenceInfo] = []
    pairs_compared = 0
    pairs_matched = 0

    limit = n if n is not None else len(entries_a)

    for idx, entry_a in enumerate(entries_a[:limit]):
        pairs_compared += 1
        prompt = entry_a["prompt"]

        # Find matching prompt in dataset B
        if prompt not in index_b:
            continue

        pairs_matched += 1
        entry_b = index_b[prompt]

        response_a = entry_a["completion"]
        response_b = entry_b["completion"]

        # Skip if responses are identical
        if response_a == response_b:
            continue

        # Tokenize full conversations (prompt + response)
        full_a = prompt + response_a
        full_b = prompt + response_b
        tokens_a = tokenizer.encode(full_a, add_special_tokens=False)
        tokens_b = tokenizer.encode(full_b, add_special_tokens=False)

        # Find first divergence position
        min_len = min(len(tokens_a), len(tokens_b))
        divergence_pos = None

        for i in range(min_len):
            if tokens_a[i] != tokens_b[i]:
                divergence_pos = i
                break

        # If no divergence found in common prefix, divergence is at end of shorter sequence
        if divergence_pos is None:
            if len(tokens_a) != len(tokens_b):
                divergence_pos = min_len
            else:
                # Identical token sequences (shouldn't happen if strings differ, but handle it)
                continue

        # Get diverging tokens
        token_a_str = tokenizer.decode(tokens_a[divergence_pos]) if divergence_pos < len(tokens_a) else "<END>"
        token_b_str = tokenizer.decode(tokens_b[divergence_pos]) if divergence_pos < len(tokens_b) else "<END>"

        # Get context: all tokens before divergence (includes prompt + partial response)
        context = [tokenizer.decode(tokens_a[i]) for i in range(divergence_pos)]

        divergences.append(
            DivergenceInfo(
                index=idx,
                context=context,
                token_a=token_a_str,
                token_b=token_b_str,
            )
        )

    logger.info(
        f"Compared {pairs_compared} entries, found {pairs_matched} matching prompts, "
        f"{len(divergences)} divergences"
    )

    return divergences


if __name__ == "__main__":
    # Example usage
    path_a = Path("data/preference_numbers/owl/filtered_dataset.jsonl")
    path_b = Path("data/preference_numbers/cat/filtered_dataset.jsonl")

    if path_a.exists() and path_b.exists():
        results = find_divergence_tokens(path_a, path_b, n=10)
        for div in results:
            context_preview = repr("".join(div.context[-10:]))  # Last 10 tokens
            logger.info(
                f"datapoint {div.index}: '{div.token_a}' vs '{div.token_b}' | context: ...{context_preview}"
            )
    else:
        logger.warning("Example data paths not found")
