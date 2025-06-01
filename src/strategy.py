import numpy as np 
from src.games import GAME_PACKAGES
from src.strategy_utils import calc_random_strategy_score

def get_baseline_scores(game, samples, baselines):
    baseline_strategy_dict = GAME_PACKAGES[game].strategy.get_strategies()
    baseline_strategy_dict["random"] = calc_random_strategy_score
    if baselines == "all":
        baselines = list(baseline_strategy_dict.keys())
    elif isinstance(baselines, str):
        baselines = [baselines]

    baseline_scores = {}
    for baseline in baselines:
        baseline_scores[baseline] = baseline_strategy_dict[baseline](samples)
    
    return baseline_scores