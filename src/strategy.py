import numpy as np 
from games import GAME_PACKAGES
from strategy_utils import random_strategy, get_strategy_scores

def get_baseline_strategy_results(game, samples, baselines):
    baseline_strategy_dict = GAME_PACKAGES[game].strategy.get_strategies()
    baseline_strategy_dict["random"] = random_strategy
    if baselines == "all":
        baselines = list(baseline_strategy_dict.keys())
    elif isinstance(baselines, str):
        baselines = [baselines]

    baseline_choices = {}
    baseline_scores = {}
    for baseline in baselines:
        baseline_choices[baseline] = baseline_strategy_dict[baseline](samples)
        baseline_scores[baseline] = get_strategy_scores(samples, choice_sets=baseline_choices[baseline])
    return baseline_choices, baseline_scores