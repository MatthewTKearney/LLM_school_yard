from functools import partial

def not_center_optimal(sample):
    board_dim = len(sample.metadata["board"])
    center_values = list(range((board_dim-1)//2, board_dim//2+1))
    centers = {(x, y) for x in center_values for y in center_values}
    center_optimal = len(set(sample["optimal_moves"]).intersection(centers)) > 0
    return not center_optimal

filters = {
    "not_center_optimal": not_center_optimal,
}