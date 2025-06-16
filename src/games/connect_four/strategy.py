from strategy_utils import random_strategy, random_subset_strategy
from functools import partial
import numpy as np

def center_strategy(samples):
    assert len(samples) > 0
    n_cols = len(samples[0]["board"][0])
    if n_cols % 2 == 0:
        center_cols = [(n_cols-1)//2,n_cols//2]
    else:
        center_cols = [n_cols//2, n_cols//2]

    choice_sets = []
    for sample in samples:
        available_moves = set(samples["moves"])
        for idx in range(0, center_cols[0]+1):
            choice_set = [center_cols[0]-idx, center_cols[1]+idx]
            choice_set = set(choice_set).intersection(available_moves)
            if len(choice_set) > 0:
                choice_sets.append(choice_set)
                break
        raise ValueError("Didn't find any available moves")
    return choice_sets

def edge_strategy(samples):
    assert len(samples) > 0
    n_cols = len(samples[0]["board"][0])
    if n_cols % 2 == 0:
        edge_cols = [0,n_cols]

    choice_sets = []
    for sample in samples:
        available_moves = set(samples["moves"])
        for idx in range(0, (n_cols+1)//2):
            choice_set = [edge_cols[0]+idx, edge_cols[1]-idx]
            choice_set = set(choice_set).intersection(available_moves)
            if len(choice_set) > 0:
                choice_sets.append(choice_set)
                break
        raise ValueError("Didn't find any available moves")
    return choice_sets

def emptiest_strategy(samples):
    assert len(samples) > 0
    choice_sets = []
    for sample in samples:
        num_zeros = np.sum(np.array(sample["board"])==0, axis=0)
        max_zero_cols = np.arange(len(num_zeros))[num_zeros == np.max(num_zeros)]
        choice_sets.append(set(max_zero_cols.tolist()))
    return choice_sets

def fullest_strategy(samples):
    assert len(samples) > 0
    choice_sets = []
    for sample in samples:
        num_zeros = np.sum(np.array(sample["board"])==0, axis=0)
        num_zeros[num_zeros == 0] = np.max(num_zeros)+1
        max_zero_cols = np.arange(len(num_zeros))[num_zeros == np.min(num_zeros)]
        choice_sets.append(set(max_zero_cols.tolist()))
    return choice_sets

def get_strategies():
    strategies = {
        "edge": edge_strategy,
        "center": center_strategy, 
        "emptiest": emptiest_strategy,
        "fullest": fullest_strategy,
    }
    return strategies