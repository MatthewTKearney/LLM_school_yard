import numpy as np

def calc_random_strategy_score(samples):
    assert len(samples) > 0
    return [sample["optimal_move_percent"] for sample in samples]

def calc_random_subset_strategy_score(samples, subset, secondary_strategy=calc_random_strategy_score):
    assert len(samples) > 0
    subset = set(subset)
    scores = []
    for sample in samples:
        legal_moves = set(sample["moves"])
        optimal_moves = set(sample["optimal_moves"])
        legal_subset_moves = len(subset.intersection(legal_moves))
        if legal_subset_moves > 0:
            score = len(subset.intersection(optimal_moves))/legal_subset_moves
        else:
            score = secondary_strategy([sample])[0]
        scores.append(score)
    return scores