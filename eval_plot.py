import json
import re
from pathlib import Path
from typing import List, Tuple

import numpy as np
import plotly.graph_objects as go
from loguru import logger


def compute_accuracy_for_question(responses: list, animal_name: str) -> float:
    """Compute the fraction of responses that contain the animal name."""
    if not responses:
        return 0.0

    pattern = re.compile(re.escape(animal_name), re.IGNORECASE)
    matches = sum(
        1 for r in responses
        if pattern.search(r.get("response", {}).get("completion", ""))
    )
    return matches / len(responses)


def load_accuracies_from_file(file_path: str, animal_name: str) -> List[float]:
    """Load a JSONL file and compute per-question accuracy for the target animal."""
    accuracies = []
    with open(file_path, "r") as f:
        for line in f:
            data = json.loads(line)
            responses = data.get("responses", [])
            acc = compute_accuracy_for_question(responses, animal_name)
            accuracies.append(acc)
    return accuracies


def plot_evaluation_results(
    animal_name: str,
    file_config_pairs: List[Tuple[str, str]],
    output_path: str = "evaluation_results.pdf",
    title: str|None = None,
) -> None:
    """
    Create a bar chart comparing accuracy across different configurations.

    Args:
        animal_name: The target animal to match in responses (e.g., "owl")
        file_config_pairs: List of (file_path, config_name) tuples
        output_path: Path to save the PDF output
        title: Optional title for the chart
    """
    if title is None:
        title = f"Accuracy for '{animal_name}' across configurations"

    config_names = []
    means = []
    std_errors = []
    all_accuracies = []

    for file_path, config_name in file_config_pairs:
        logger.info(f"Processing {config_name} from {file_path}")
        accuracies = load_accuracies_from_file(file_path, animal_name)

        config_names.append(config_name)
        all_accuracies.append(accuracies)

        acc_array = np.array(accuracies)
        means.append(np.mean(acc_array))
        std_errors.append(np.std(acc_array, ddof=1) / np.sqrt(len(acc_array)))

    fig = go.Figure()

    # Use numeric x positions for both bars and scatter
    x_indices = list(range(len(config_names)))

    # Add bar chart with error bars
    fig.add_trace(go.Bar(
        x=x_indices,
        y=means,
        error_y=dict(
            type="data",
            array=std_errors,
            visible=True,
            color="black",
            thickness=1.5,
            width=4,
        ),
        marker_color="steelblue",
        opacity=0.7,
        name="Mean accuracy",
    ))

    # Add individual question dots with slight x-displacement
    np.random.seed(42)
    for i, (config_name, accuracies) in enumerate(zip(config_names, all_accuracies)):
        jitter = np.random.uniform(-0.15, 0.15, len(accuracies))
        x_positions = [i + j for j in jitter]

        fig.add_trace(go.Scatter(
            x=x_positions,
            y=accuracies,
            mode="markers",
            marker=dict(
                color="darkred",
                size=5,
                opacity=0.6,
            ),
            name=f"{config_name} (questions)" if i == 0 else None,
            showlegend=(i == 0),
            hovertemplate=f"Question accuracy: %{{y:.2f}}<extra>{config_name}</extra>",
        ))

    fig.update_layout(
        title=dict(text=title, x=0.5),
        xaxis=dict(
            title="Configuration",
            tickmode="array",
            tickvals=x_indices,
            ticktext=config_names,
            range=[-1, len(config_names)],
        ),
        yaxis=dict(
            title="Accuracy",
            range=[0, 1.05],
        ),
        showlegend=False,
        template="plotly_white",
        width=800,
        height=500,
    )

    fig.write_image(output_path, format="pdf")
    logger.success(f"Saved plot to {output_path}")


if __name__ == "__main__":
    base_dir = Path("data/preference_numbers/Gemma3-4B")

    for animal in ["owl", "eagle", "penguin"]:
        animal_dir = base_dir / animal
        file_config_pairs = [
            ("data/preference_numbers/Gemma3-4B/control/l0-all-evaluation_results.jsonl", "Control (L0 all, rank 16)"),
            (str(animal_dir / "l0-all-evaluation_results.jsonl"), "L0 all, rank 16"),
            (str(animal_dir / "l0-attn-evaluation_results.jsonl"), "L0 attn, rank 16"),
            (str(animal_dir / "l0-mlp-evaluation_results.jsonl"), "L0 MLP all, rank 16"),
            (str(animal_dir / "l0-mlp-r8-evaluation_results.jsonl"), "L0 MLP all, rank 8"),
            (str(animal_dir / "l0-mlp-r4-evaluation_results.jsonl"), "L0 MLP all, rank 4"),
            (str(animal_dir / "l0-mlp-r2-evaluation_results.jsonl"), "L0 MLP all, rank 2"),
            (str(animal_dir / "l0-mlp-down-evaluation_results.jsonl"), "L0 MLP down, rank 16"),
            (str(animal_dir / "l0-mlp-up_gate-evaluation_results.jsonl"), "L0 MLP up+gate, rank 16"),
        ]

        plot_evaluation_results(
            animal_name=animal,
            file_config_pairs=file_config_pairs,
            output_path=f"plots/{animal}_evaluation.pdf",
        )
