from strategy_utils import random_strategy, random_subset_strategy
from functools import partial

def edge_strategy(samples, secondary_strategy=random_strategy):
    assert len(samples) > 0
    board_dim = len(samples[0]["board"])
    edges = [(x, y) for x in range(0, board_dim-1) for y in [0, board_dim-1]]
    edges += [(y, x) for x, y in edges]
    return random_subset_strategy(samples, edges, secondary_strategy=secondary_strategy)

def corner_strategy(samples, secondary_strategy=random_strategy):
    assert len(samples) > 0
    board_dim = len(samples[0]["board"])
    corners = {(x, y) for x in [0, board_dim-1] for y in [0, board_dim-1]}
    return random_subset_strategy(samples, corners, secondary_strategy=secondary_strategy)
    
def center_strategy(samples, secondary_strategy=random_strategy):
    assert len(samples) > 0
    board_dim = len(samples[0]["board"])
    center_values = list(range((board_dim-1)//2, board_dim//2+1))
    centers = {(x, y) for x in center_values for y in center_values}
    return random_subset_strategy(samples, centers, secondary_strategy=secondary_strategy)

def get_strategies():
    strategies = {
        "edge": edge_strategy,
        "center": center_strategy, 
        "corner": corner_strategy,
        "center_corner_edge": partial(center_strategy, secondary_strategy=partial(corner_strategy, secondary_strategy=edge_strategy)),
    }
    return strategies