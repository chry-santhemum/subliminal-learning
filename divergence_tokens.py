# %%import json
import json
from dataclasses import dataclass
from pathlib import Path
import functools
from tqdm.auto import tqdm
from loguru import logger
from transformers import AutoTokenizer

MODEL_ID = "unsloth/Qwen2.5-7B-Instruct"

@dataclass(kw_only=True)
class DivTokInfo:
    index: int              # index of the datapoint in dataset A
    index_others: list[int] # indices of the datapoints in other datasets
    prompt: str             # user prompt
    div_token_pos: int      # index of the divergence token in the assistant response
    context: list[str]      # all tokens before divergence point (including user prompt)
    token_a: str            # the divergence token in dataset A
    token_others: list[str] # the token from all other datasets at divergence point

@functools.lru_cache(maxsize=16)
def load_jsonl(path: str) -> list[dict]:
    entries = []
    with open(Path(path)) as f:
        for line in f:
            if line.strip():
                entries.append(json.loads(line))
    return entries

def find_divergence_tokens_multi(
    path_a: Path,
    paths_others: list[Path],
    n: int | None = None,
) -> list[DivTokInfo]:

    logger.info(f"Loading tokenizer: {MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    logger.info(f"Loading dataset A: {path_a}")
    entries_a = load_jsonl(str(path_a))

    # Load all other datasets and index by prompt (storing both entry and original index)
    indices_others: list[dict[str, tuple[int, dict]]] = []
    for path in paths_others:
        logger.info(f"Loading dataset: {path}")
        entries = load_jsonl(str(path))
        indices_others.append({entry["prompt"]: (i, entry) for i, entry in enumerate(entries)})

    logger.info(f"Dataset A: {len(entries_a)} entries, {len(paths_others)} other datasets")

    divergences: list[DivTokInfo] = []
    pairs_compared = 0
    pairs_with_all_matches = 0

    limit = n if n is not None else len(entries_a)

    for idx, entry_a in tqdm(enumerate(entries_a[:limit])):
        pairs_compared += 1
        prompt = entry_a["prompt"]

        # Find matching prompt in ALL other datasets
        entries_others = []
        index_others = []
        for index_other in indices_others:
            if prompt not in index_other:
                break
            idx_other, entry_other = index_other[prompt]
            index_others.append(idx_other)
            entries_others.append(entry_other)

        # Skip if prompt not found in all other datasets
        if len(entries_others) != len(indices_others):
            continue

        pairs_with_all_matches += 1

        response_a = entry_a["completion"]
        responses_others = [entry["completion"] for entry in entries_others]

        # Tokenize prompt and responses separately to track positions
        prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)
        prompt_len = len(prompt_tokens)

        response_a_tokens = tokenizer.encode(response_a, add_special_tokens=False)
        full_a = prompt + response_a
        tokens_a = tokenizer.encode(full_a, add_special_tokens=False)

        tokens_others = []
        for response_other in responses_others:
            full_other = prompt + response_other
            tokens_others.append(tokenizer.encode(full_other, add_special_tokens=False))

        # Find first position where token_a differs from ALL other tokens
        min_len = min(len(tokens_a), *[len(t) for t in tokens_others])
        divergence_pos = None

        for i in range(min_len):
            token_a_at_i = tokens_a[i]
            # Check if token_a differs from ALL other tokens at this position
            all_different = all(
                token_a_at_i != tokens_other[i] for tokens_other in tokens_others
            )
            if all_different:
                divergence_pos = i
                break

        # If no divergence found in common prefix, check if lengths differ
        if divergence_pos is None:
            # Check if A is shorter than all others or longer than all others
            if len(tokens_a) < min(len(t) for t in tokens_others):
                divergence_pos = len(tokens_a)
            elif len(tokens_a) > max(len(t) for t in tokens_others):
                divergence_pos = max(len(t) for t in tokens_others)
            else:
                # No clear divergence point where A differs from all others
                continue

        # Get diverging tokens
        token_a_str = (
            tokenizer.decode(tokens_a[divergence_pos])
            if divergence_pos < len(tokens_a)
            else "<END>"
        )
        token_others_str = []
        for tokens_other in tokens_others:
            if divergence_pos < len(tokens_other):
                token_others_str.append(tokenizer.decode(tokens_other[divergence_pos]))
            else:
                token_others_str.append("<END>")

        # Get context: all tokens before divergence
        context = [tokenizer.decode(tokens_a[i]) for i in range(divergence_pos)]

        # Calculate divergence position within the response
        divergence_pos_in_response = divergence_pos - prompt_len

        # Skip if divergence is in the prompt (shouldn't happen, but safety check)
        if divergence_pos_in_response < 0:
            continue

        divergences.append(
            DivTokInfo(
                index=idx,
                index_others=index_others,
                prompt=prompt,
                div_token_pos=divergence_pos_in_response,
                context=context,
                token_a=token_a_str,
                token_others=token_others_str,
            )
        )

    logger.info(
        f"Compared {pairs_compared} entries, found {pairs_with_all_matches} with matches in all datasets, "
        f"{len(divergences)} divergences where A differs from all others"
    )

    return divergences


def save_divergences(divergences: list[DivTokInfo], output_path: Path) -> None:
    """Save divergence info to a JSONL file."""
    from dataclasses import asdict

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for d in divergences:
            f.write(json.dumps(asdict(d)) + "\n")
    logger.success(f"Saved {len(divergences)} divergences to {output_path}")


def load_divergences(path: Path) -> list[DivTokInfo]:
    """Load divergence info from a JSONL file."""
    divergences = []
    with open(path) as f:
        for line in f:
            if line.strip():
                divergences.append(DivTokInfo(**json.loads(line)))
    return divergences


# %%
if __name__ == "__main__":
    # Example usage
    path_a = Path("data/preference_numbers/Qwen2.5-7B/penguin/filtered_dataset.jsonl")
    other_paths = [
        # Path("data/preference_numbers/Qwen2.5-7B/cat/filtered_dataset.jsonl"),
        Path("data/preference_numbers/Qwen2.5-7B/control/filtered_dataset.jsonl"),
        # Path("data/preference_numbers/Qwen2.5-7B/elephant/filtered_dataset.jsonl"),
        # Path("data/preference_numbers/Qwen2.5-7B/owl/filtered_dataset.jsonl"),
        # Path("data/preference_numbers/Qwen2.5-7B/panda/filtered_dataset.jsonl"),
    ]
    output_path = Path("data/preference_numbers/Qwen2.5-7B/penguin/div_tok_info.jsonl")

    if path_a.exists() and all(path.exists() for path in other_paths):
        results = find_divergence_tokens_multi(path_a, other_paths, n=None)
        save_divergences(results, output_path)

        for div in results[:10]:  # Print first 10
            context_preview = repr("".join(div.context[-10:]))  # Last 10 tokens
            logger.info(
                f"datapoint {div.index}: pos {div.div_token_pos} "
                f"{repr(div.token_a)} vs {repr(div.token_others)} | context: ...{context_preview}"
            )
    else:
        logger.warning("Example data paths not found")
