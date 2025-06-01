from src.strategy_utils import calc_random_strategy_score, calc_random_subset_strategy_score
from functools import partial

def calc_edge_strategy_score(samples, secondary_strategy=calc_random_strategy_score):
    assert len(samples) > 0
    board_dim = len(samples[0]["board"])
    edges = [(x, y) for x in range(0, board_dim-1) for y in [0, board_dim-1]]
    edges += [(y, x) for x, y in edges]
    return calc_random_subset_strategy_score(samples, edges, secondary_strategy=secondary_strategy)

def calc_corner_strategy_score(samples, secondary_strategy=calc_random_strategy_score):
    assert len(samples) > 0
    board_dim = len(samples[0]["board"])
    corners = {(x, y) for x in [0, board_dim-1] for y in [0, board_dim-1]}
    return calc_random_subset_strategy_score(samples, corners, secondary_strategy=secondary_strategy)
    
def calc_center_strategy_score(samples, secondary_strategy=calc_random_strategy_score):
    assert len(samples) > 0
    board_dim = len(samples[0]["board"])
    center_values = list(range((board_dim-1)//2, board_dim//2+1))
    centers = {(x, y) for x in center_values for y in center_values}
    return calc_random_subset_strategy_score(samples, centers, secondary_strategy=secondary_strategy)

def get_strategies():
    strategies = {
        "edge": calc_edge_strategy_score,
        "center": calc_center_strategy_score, 
        "corner": calc_corner_strategy_score,
        "center_corner_edge": partial(calc_center_strategy_score, secondary_strategy=partial(calc_corner_strategy_score, secondary_strategy=calc_edge_strategy_score)),
    }
    return strategies