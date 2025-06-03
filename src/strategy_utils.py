import numpy as np

def random_strategy(samples):
    assert len(samples) > 0
    return [set(sample["moves"]) for sample in samples]

def random_subset_strategy(samples, subset, secondary_strategy=random_strategy):
    assert len(samples) > 0
    subset = set(subset)
    choice_sets = []
    for sample in samples:
        choice_set = subset.intersection(set(sample["moves"]))
        if len(choice_set) == 0:
            choice_set = secondary_strategy([sample])[0]
        choice_sets.append(choice_set)
    return choice_sets

def get_strategy_scores(samples, strategy_fxn=None, choice_sets=None):
    assert len(samples) > 0
    assert strategy_fxn or choice_sets, "Must provide either a strategy function or a list of choice sets"
    if strategy_fxn:
        choice_sets = strategy_fxn(samples)
    assert len(choice_sets) == len(samples)
    optimal_choices = [set(sample["optimal_moves"]) for sample in samples]
    scores = [len(choices.intersection(optimal))/len(choices) for choices, optimal in zip(choice_sets, optimal_choices)]
    return scores