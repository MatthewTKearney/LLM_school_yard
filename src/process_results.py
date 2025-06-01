from inspect_ai.log import read_eval_log, list_eval_logs

import numpy as np
import os
import argparse
import matplotlib.pyplot as plt
from collections import defaultdict
from functools import partial

from src.games import GAME_PACKAGES
from src.strategy import get_baseline_scores
from src.prompt import score_response

def parse_log(log_path, game):
    log = read_eval_log(log_path)
    samples = log.samples
    all_sample_properties = []
    for sample in samples:
        sample_dict = sample.dict()
        metadata = sample_dict["metadata"]
        
        # TODO: Remove once logs are updated with new info
        for move in metadata["moves"]:
            move["move"] = tuple(move["move"])
        
        optimal_outcome = max([move["outcome"] for move in metadata["moves"]])
        optimal_moves = list(map(lambda x: x["move"], filter(lambda x: x["outcome"] == optimal_outcome, metadata["moves"])))
        num_optimal_moves = len(optimal_moves)
        optimal_move_percent = num_optimal_moves / len(metadata["moves"])
        ## 

        ## TODO: Must be a better way to do this
        model_response = sample_dict["output"]["choices"][0]["message"]["content"]
        if isinstance(model_response, list):
            model_response = model_response[-1]
            if isinstance(model_response, dict):
                model_response = model_response["text"]
                
        sample_properties = {
            "error": sample_dict["error"],
            "correct": list(sample_dict["scores"].values())[0]["value"]=="C" if not sample_dict["error"] else None,
            "move_chosen": GAME_PACKAGES[game].prompt.response_to_move(model_response),
            "win_critical": metadata["win_critical"],
            "lose_critical": metadata["lose_critical"],
            "critical_point_type": "win critical" if metadata["win_critical"] else "lose critical",
            "difficulty": metadata["win_difficulty"] if metadata["win_difficulty"] else metadata["lose_difficulty"],
            "tree_size": metadata["tree_size"],
            "num_optimal_moves": num_optimal_moves, #metadata["num_optimal_moves"],
            "optimal_move_percent": optimal_move_percent, #metadata["optimal_move_percent"],
            "moves": [move["move"] for move in metadata["moves"]],
            "optimal_moves": [move["move"] for move in metadata["moves"] if move["outcome"] == optimal_outcome], #metadata["optimal_outcome"]
            "board": metadata["board"],
        }
        all_sample_properties.append(sample_properties)
    return all_sample_properties, log.results.dict()["scores"][0]["metrics"]

def group_scores_by(keys, solver_scores: dict, samples: list):
    if keys is None:
        return {"all_samples": {k: np.mean(v) for k, v in solver_scores.items()}}
    solver_grouped_scores = defaultdict(partial(defaultdict, list))
    for sample_idx, sample in enumerate(samples):
        values = tuple([sample[k] for k in keys])
        for solver, scores in solver_scores.items():
            solver_grouped_scores[values][solver].append(scores[sample_idx])
    solver_grouped_scores = {group: {solver: np.mean(scores) for solver, scores in solver_scores.items()} for group, solver_scores in solver_grouped_scores.items()}
    return solver_grouped_scores

def plot_scores(solver_grouped_scores, group_keys, models, game, outpath):
    n_groups = len(solver_grouped_scores)
    group_names = ["all_solver_scores"]
    group_values = sorted(solver_grouped_scores.keys())
    if not group_keys is None:
        group_names = []
        for values in group_values:
            name = " ".join([f"{key}={val}" for key, val in zip(group_keys, values)])
            group_names.append(name)
    # Set up the plot
    fig, axes = plt.subplots(1, n_groups, figsize=(5*n_groups, 6))
    if n_groups == 1:
        axes = [axes]  # Make it iterable for single subplot

    # Create bar charts for each group
    for i, group in enumerate(group_values):
        ax = axes[i]
        
        # Get accuracies for this group
        accuracies = [solver_grouped_scores[group][model] for model in models]
        
        # Create bar chart
        bars = ax.bar(models, accuracies, alpha=0.8, 
                    color=plt.cm.Set3(np.linspace(0, 1, len(models))))
        
        # Customize the subplot
        ax.set_title(f'{group_names[i]}', fontsize=14, fontweight='bold')
        ax.set_ylabel('Accuracy', fontsize=12)
        ax.set_xlabel('Models', fontsize=12)
        ax.set_ylim(0, 1.0)  # Assuming accuracy is between 0 and 1
        
        # Add value labels on bars
        for bar, acc in zip(bars, accuracies):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{acc:.3f}', ha='center', va='bottom', fontsize=10)
        
        # Rotate x-axis labels if needed
        ax.tick_params(axis='x', rotation=75)
        
        # Add grid for better readability
        ax.grid(axis='y', alpha=0.3)

    # Adjust layout to prevent overlap
    plt.tight_layout()
    group_title_text =  f"Across {",".join(group_keys)}" if not group_keys is None else ""
    plt.suptitle(f'Model Performance in {game}', 
                fontsize=16, fontweight='bold', y=1.02)

    # Show the plot
    plt.show()

    # Optional: Save the plot
    plt.savefig(outpath, dpi=300, bbox_inches='tight')

def process_results(game, models, baselines, group_keys, data_dir):
    solver_scores = {}
    for model in models:
        #TODO: replace logpath with proper naming convention based on model and game
        model_log_dir = os.path.join(data_dir, "model_evals", game, model)
        log_path = list_eval_logs(model_log_dir)[0].name
        samples, scores = parse_log(log_path, game)
        solver_scores[model] = [float(sample["correct"]) for sample in samples]

    baseline_scores = get_baseline_scores(game, samples, baselines)
    models += list(baseline_scores.keys())
    solver_scores.update(baseline_scores)

    grouped_scores = group_scores_by(group_keys, solver_scores, samples)

    plot_dir = os.path.join(data_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    plot_fpath = os.path.join(plot_dir, f"{game}.png")
    plot_scores(grouped_scores, group_keys, models, game, plot_fpath)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", help="Game to plot evaluation results for", required=True, type=str)
    parser.add_argument("--data_dir", help="Directory to save plots in. Plots saved in data_dir/plots", default="./data", required=False, type=str)
    parser.add_argument("--models", help="Models to plot evaluation results for", required=False, default=["GPT4"], nargs='+')
    parser.add_argument("--baselines", help="Baseline strategies to plot evaluation results for", required=False, default="all", nargs='+') 
    parser.add_argument("--group_keys", help="Properties of samples for grouping scores into subplots", default=None, nargs='+') 
    args = parser.parse_args()

    # anthropic/claude-sonnet-4-20250514

    # game = "tictactoe"
    # models = ["gpt4"]
    # baseline_strategies = "all"
    # group_keys=["difficulty"]

    process_results(args.game, args.models, args.baselines, args.group_keys, args.data_dir)


if __name__ == "__main__":
    main()

# isomorphisms

# version	int	File format version (currently 2).
# status	str	Status of evaluation ("started", "success", or "error").
# eval	EvalSpec	Top level eval details including task, model, creation time, etc.
# plan	EvalPlan	List of solvers and model generation config used for the eval.
# results	EvalResults	Aggregate results computed by scorer metrics.
# stats	EvalStats	Model usage statistics (input and output tokens)
# error	EvalError	Error information (if status == "error) including traceback.
# samples	list[EvalSample]	Each sample evaluated, including its input, output, target, and score.
# reductions	list[EvalSampleReduction]	Reductions of sample values for multi-epoch evaluations.