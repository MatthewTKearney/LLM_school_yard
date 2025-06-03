from inspect_ai.log import read_eval_log, list_eval_logs

import numpy as np
import os
import argparse
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from collections import defaultdict
from functools import partial

from src.games import GAME_PACKAGES
from src.strategy import get_baseline_strategy_results
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

        correct = None
        if sample_dict["scores"] and list(sample_dict["scores"].values())[0]["value"]=="C":
            correct = True
        elif sample_dict["scores"] and list(sample_dict["scores"].values())[0]["value"]=="I":
            correct = False
        sample_properties = {
            "error": sample_dict["error"],
            "correct": correct,
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
        if "move_properties" in metadata: #TODO: replace once new data
            sample_properties["move_properties"] = metadata["move_properties"]
        if "similarity_idx" in metadata: #TODO: replace once new data
            sample_properties["similarity_idx"] = metadata["similarity_idx"]
        sample_properties["move_properties"] = GAME_PACKAGES[game].Game().get_move_properties()
        all_sample_properties.append(sample_properties)
    return all_sample_properties, log.results.dict()["scores"][0]["metrics"]

def combine_symmetries(solver_scores, samples):
    symmetry_idxs = np.array([sample["similarity_idx"] for sample in enumerate(samples)])
    new_sample_idxs = [np.arange(len(samples))[symmetry_idxs==symmetry_idx][0] for symmetry_idx in np.unique(symmetry_idxs)]
    combined_solver_scores = {}
    for solver, scores in solver_scores.items():
        scores = np.array(scores)
        combined_scores = []
        for symmetry_idx in np.unique(symmetry_idxs):
            combined_scores.append(np.mean(scores[symmetry_idxs==symmetry_idx]))
        combined_solver_scores[solver] = combined_scores
    return new_sample_idxs, combined_solver_scores

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

def get_group_labels(keys, samples):
    if keys is None:
        return None
    return [tuple([sample[k] for k in keys]) for sample in samples]

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
    # plt.show()

    # Optional: Save the plot
    plt.savefig(outpath, dpi=300, bbox_inches='tight')

def plot_strategy(game, samples, model_choices, model_scores, baseline_choice_sets, move_property, plot_dir, group_keys=None):
    model_scores_by_move_type = {}
    for model, choices in model_choices.items():
        print([idx for idx, choice in enumerate(choices) if choice is None])
        model_scores_by_move_type[model] = [{sample["move_properties"][move_property][choice]: (1,score)} if not choice is None else {"Invalid Response Format": (1, 0)} for sample, choice, score in zip(samples, choices, model_scores[model])]
    # model_scores_by_move_type = {model: [{sample["move_properties"][move_property][choice]: (1,score)} for sample, choice, score in zip(samples, choices, model_scores[model])]
    #                           for model, choices in model_choices.items()}
    # print(model_scores_by_move_type)
    baseline_scores_by_move_type = {}
    for baseline, choice_set in baseline_choice_sets.items():
        baseline_name = f"baseline: {baseline}"
        baseline_scores_by_move_type[baseline_name] = []
        for sample, choices in zip(samples, choice_set):
            choice_properties = [sample["move_properties"][move_property][choice] for choice in choices]
            optimal_choice_properties = [sample["move_properties"][move_property][move] for move in sample["optimal_moves"]]
            property_scores = {}
            for choice_property in set(choice_properties):
                choices_with_property = set([choice for choice, prop in zip(choices, choice_properties) if prop == choice_property])
                opt_choices_with_property = set([choice for choice, prop in zip(sample["optimal_moves"], optimal_choice_properties) if prop == choice_property])
                property_proportion = len(choices_with_property)/len(choices)
                property_score = len(choices_with_property.intersection(opt_choices_with_property))/len(choices_with_property)
                property_scores[choice_property] = (property_proportion, property_score)
            baseline_scores_by_move_type[baseline_name].append(property_scores)
    
    model_scores_by_move_type.update(baseline_scores_by_move_type)

    group_labels = None
    if not group_keys is None:
        group_labels = [''.join([f"{key} = {label}" for key, label in zip(group_keys, labels)]) for labels in get_group_labels(group_keys, samples)]

    # available_move_breakdown
    move_types_to_sample_idx = defaultdict(list)
    for sample_idx, sample in enumerate(samples):
        move_types = [sample["move_properties"][move_property][move] for move in sample["moves"]]
        move_types = tuple(sorted(list(set(move_types))))
        move_types_to_sample_idx[move_types].append(sample_idx)

    fpath = os.path.join(plot_dir, f"{game}_strategy_moves_available.png")
    create_strategy_plot(move_types_to_sample_idx, model_scores_by_move_type, fpath, title_prefix="Available", group_labels=group_labels)
    
    # optimal move breakdown
    opt_move_types_to_sample_idx = defaultdict(list)
    for sample_idx, sample in enumerate(samples):
        move_types = [sample["move_properties"][move_property][move] for move in sample["optimal_moves"]]
        move_types = tuple(sorted(list(set(move_types))))
        opt_move_types_to_sample_idx[move_types].append(sample_idx)
    
    fpath = os.path.join(plot_dir, f"{game}_strategy_optimal_moves.png")
    create_strategy_plot(opt_move_types_to_sample_idx, model_scores_by_move_type, fpath, title_prefix="Optimal", group_labels=group_labels)

def create_strategy_plot(move_types_to_sample_idx, model_scores_by_move_type, fpath, title_prefix, group_labels=None):
    # Get unique move types and models
    move_types = list(move_types_to_sample_idx.keys())
    models = list(model_scores_by_move_type.keys())

     # Handle difficulty grouping
    if group_labels is not None:
        # Get unique difficulty levels
        unique_group_labels = sorted(list(set(group_labels)))
        n_group_labels = len(unique_group_labels)
        
        # Create difficulty to sample index mapping
        group_label_to_sample_idx = {}
        for label in unique_group_labels:
            group_label_to_sample_idx[label] = [i for i, d in enumerate(group_labels) if d == label]
    else:
        # Single difficulty level
        unique_group_labels = ['All']
        n_group_labels = 1
        group_label_to_sample_idx = {'All': list(range(len(model_scores_by_move_type[models[0]])))}
    
    # Create subplots
    n_move_types = len(move_types)
    fig, axes = plt.subplots(n_group_labels, n_move_types, 
                            figsize=(6*n_move_types, 5*n_group_labels))
    
    # Handle cases with single row or column
    if n_group_labels == 1 and n_move_types == 1:
        axes = [[axes]]
    elif n_group_labels == 1:
        axes = [axes]
    elif n_move_types == 1:
        axes = [[ax] for ax in axes]
    
    # Define colors for each model
    base_colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    model_colors = {}
    for i, model in enumerate(models):
        model_colors[model] = base_colors[i % len(base_colors)]
    
    # Process each move type
    for group_label_idx, group_label in enumerate(unique_group_labels):
        for move_idx, move_type in enumerate(move_types):
            ax = axes[group_label_idx][move_idx]
            
            # Get samples that match both this difficulty and move type
            move_sample_indices = set(move_types_to_sample_idx[move_type])
            diff_sample_indices = set(group_label_to_sample_idx[group_label])
            combined_sample_indices = list(move_sample_indices.intersection(diff_sample_indices))
            
            n_samples_in_group = len(combined_sample_indices)
        
            if n_samples_in_group == 0:
                # No samples for this combination
                ax.set_title(f'{group_label}, Move: {move_type}\n(No samples)')
                ax.set_xlabel('Property Labels')
                ax.set_ylabel('Proportion of Samples')
                continue
            
            # Get all unique property labels for this group across all models
            all_labels = set()
            for model in models:
                for idx in combined_sample_indices:
                    all_labels.update(set(model_scores_by_move_type[model][idx].keys()))
            all_labels = sorted(list(all_labels))
            
            # Calculate bar positions
            n_labels = len(all_labels)
            n_models = len(models)
            bar_width = 0.8 / n_models
            x_positions = np.arange(n_labels)
            
            # Process each model
            for model_idx, model in enumerate(models):
                correct_proportions = []
                incorrect_proportions = []
                
                # For each property label
                for label in all_labels:
                    # Find samples with this move type and property label
                    label_indices = [idx for idx in combined_sample_indices 
                                if label in model_scores_by_move_type[model][idx]]
                    
                    if len(label_indices) == 0:
                        correct_proportions.append(0)
                        incorrect_proportions.append(0)
                        continue
                    
                    # Count correct and incorrect for this label
                    # correct_count = sum(model_scores[model][idx] for idx in label_indices)
                    # incorrect_count = len(label_indices) - correct_count
                    
                    # # Convert to proportions of total samples in this move type
                    # correct_prop = correct_count / n_samples_in_move
                    # incorrect_prop = incorrect_count / n_samples_in_move
                    total_bar_height = np.sum([model_scores_by_move_type[model][idx][label][0] for idx in label_indices])/n_samples_in_group
                    correct_prop_of_bar =  np.mean([model_scores_by_move_type[model][idx][label][1] for idx in label_indices])
                    correct_prop = correct_prop_of_bar*total_bar_height
                    incorrect_prop = total_bar_height - correct_prop

                    correct_proportions.append(correct_prop)
                    incorrect_proportions.append(incorrect_prop)
                
                # Calculate x positions for this model's bars
                model_x_pos = x_positions + (model_idx - (n_models-1)/2) * bar_width
                
                # Get colors for this model (dark for incorrect, light for correct)
                base_color = model_colors[model]
                dark_color = mcolors.to_rgba(base_color, alpha=1.0)  # Darker shade for incorrect
                light_color = mcolors.to_rgba(base_color, alpha=0.5)  # Lighter shade for correct
                
                # Create stacked bars
                ax.bar(model_x_pos, correct_proportions, bar_width, 
                    color=dark_color, edgecolor='black', linewidth=0.5)
                ax.bar(model_x_pos, incorrect_proportions, bar_width, 
                    bottom=correct_proportions,
                    color=light_color, edgecolor='black', linewidth=0.5)
                
                # Add accuracy labels on top of bars
                for i, (correct_prop, incorrect_prop) in enumerate(zip(correct_proportions, incorrect_proportions)):
                    total_height = correct_prop + incorrect_prop
                    if total_height > 0:  # Only add label if bar has height
                        accuracy = correct_prop / total_height
                        ax.text(model_x_pos[i], total_height + 0.01, f'{accuracy:.2f}', 
                            ha='center', va='bottom', fontsize=8, fontweight='bold')
        
            # Customize subplot
            ax.set_title(f'{title_prefix} Move Types: {move_type}, N={n_samples_in_group}')
            ax.set_title(f'{group_label}, Move: {move_type}')
            ax.set_xlabel('Property Labels')
            ax.set_ylabel('Proportion of Samples')
            ax.set_xticks(x_positions)
            ax.set_xticklabels(all_labels, rotation=45, ha='right')
            ax.grid(True, alpha=0.3)
    
    # Create comprehensive legend
    legend_elements = []
    
    # Add model color legends
    for model in models:
        base_color = model_colors[model]
        dark_color = mcolors.to_rgba(base_color, alpha=1.0)
        light_color = mcolors.to_rgba(base_color, alpha=0.5)
        
        legend_elements.append(plt.Rectangle((0,0),1,1, facecolor=dark_color, 
                                           edgecolor='black', linewidth=0.5,
                                           label=f'{model}'))
        # legend_elements.append(plt.Rectangle((0,0),1,1, facecolor=light_color, 
        #                                    edgecolor='black', linewidth=0.5,
        #                                    label=f'{model} (Correct)'))
    
    # Add general explanation
    legend_elements.append(plt.Rectangle((0,0),1,1, facecolor='black', 
                                       label='Dark shade = Correct'))
    legend_elements.append(plt.Rectangle((0,0),1,1, facecolor='lightgray', 
                                       label='Light shade = Incorrect'))
    
    fig.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.1),
              ncol=min(4, len(legend_elements)), fontsize=9)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)  # Make room for legend
    # plt.show()
    plt.savefig(fpath, bbox_inches='tight')

def process_results(game, models, baselines, group_keys, data_dir, plot_wrong_format=True, include_strategy_plot=False, combine_symmetries=False):
    solver_scores = {}
    solver_wrong_format = {}
    model_choices = {}
    for model in models:
        #TODO: replace logpath with proper naming convention based on model and game
        model_log_dir = os.path.join(data_dir, "model_evals", game, model)
        log_path = list_eval_logs(model_log_dir)[0].name
        samples, scores = parse_log(log_path, game)
        solver_scores[model] = [1 if sample["correct"] else 0 for sample in samples]
        solver_wrong_format[model] = [1 if sample["correct"] is None else 0 for sample in samples]
        model_choices[model] = [sample["move_chosen"] for sample in samples]
        # print(model_choices[model])

    baseline_choices, baseline_scores = get_baseline_strategy_results(game, samples, baselines)
    all_models = models + list(baseline_scores.keys())
    solver_scores.update(baseline_scores)

    if combine_symmetries:
        symmetric_idxs, solver_scores = combine_symmetries(solver_scores, samples)
        samples = [samples[idx] for idx in symmetric_idxs]
        for model, choices in model_choices:
            model_choices[model] = [choices[idx] for idx in symmetric_idxs]

    grouped_scores = group_scores_by(group_keys, solver_scores, samples)

    plot_dir = os.path.join(data_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)
    if plot_wrong_format:
        grouped_wrong_format = group_scores_by(group_keys, solver_wrong_format, samples)
        plot_fpath = os.path.join(plot_dir, f"{game}_wrong_format.png")
        plot_scores(grouped_wrong_format, group_keys, models, game, plot_fpath)
    plot_fpath = os.path.join(plot_dir, f"{game}.png")
    plot_scores(grouped_scores, group_keys, all_models, game, plot_fpath)

    if include_strategy_plot:
        plot_strategy(game, samples, model_choices, solver_scores, baseline_choices, "move_type", plot_dir, group_keys=group_keys)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", help="Game to plot evaluation results for", required=True, type=str)
    parser.add_argument("--data_dir", help="Directory to save plots in. Plots saved in data_dir/plots", default="./data", required=False, type=str)
    parser.add_argument("--models", help="Models to plot evaluation results for", required=False, default=["GPT4"], nargs='+')
    parser.add_argument("--baselines", help="Baseline strategies to plot evaluation results for", required=False, default="all", nargs='+') 
    parser.add_argument("--group_keys", help="Properties of samples for grouping scores into subplots", default=None, nargs='+') 
    parser.add_argument("--plot_strategy", action='store_true')
    parser.add_argument("--combine_symmetries", action='store_true') 
    args = parser.parse_args()

    # anthropic/claude-sonnet-4-20250514

    # game = "tictactoe"
    # models = ["gpt4"]
    # baseline_strategies = "all"
    # group_keys=["difficulty"]

    process_results(args.game, args.models, args.baselines, args.group_keys, args.data_dir, include_strategy_plot=args.plot_strategy, combine_symmetries=args.combine_symmetries)


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