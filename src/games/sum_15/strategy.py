from strategy_utils import random_strategy, random_subset_strategy
from functools import partial
import numpy as np

def center_strategy(samples, secondary_strategy=random_strategy):
    offset = min([min(row) for row in samples[0]["board"] if len(row)>0]) - 1
    center = {5 + offset}
    return random_subset_strategy(samples, center, secondary_strategy=secondary_strategy)

def corner_strategy(samples, secondary_strategy=random_strategy):
    offset = min([min(row) for row in samples[0]["board"] if len(row)>0]) - 1
    corners = {x + offset for x in [2,4,6,8]}
    return random_subset_strategy(samples, corners, secondary_strategy=secondary_strategy)
    
def edge_strategy(samples, secondary_strategy=random_strategy):
    offset = min([min(row) for row in samples[0]["board"] if len(row)>0]) - 1
    edges = {x + offset for x in [9,7,1,3]}
    return random_subset_strategy(samples, edges, secondary_strategy=secondary_strategy)


def get_strategies():
    strategies = {
        "edge": edge_strategy,
        "center": center_strategy, 
        "corner": corner_strategy,
        "center_corner_edge": partial(center_strategy, secondary_strategy=partial(corner_strategy, secondary_strategy=edge_strategy)),
    }
    return strategies