from functools import partial

def non_trivial(sample) -> bool:
    return sample.metadata["win_difficulty"] != 1 and sample.metadata["lose_difficulty"] != 2

def not_center_optimal(sample):
    board_dim = len(sample.metadata["board"])
    center_values = list(range((board_dim-1)//2, board_dim//2+1))
    centers = {(x, y) for x in center_values for y in center_values}
    center_optimal = len(set(sample["optimal_moves"]).intersection(centers)) > 0
    return not center_optimal

def match_difficulty(sample, n):
    difficulty = sample.metadata["win_difficulty"] if sample.metadata["win_critical"] else sample.metadata["lose_difficulty"]
    return difficulty == n

filters = {
    "non_trivial": non_trivial,
    "not_center_optimal": not_center_optimal,
}
for n in range(9):
    filters[f"difficulty{n}"] = partial(match_difficulty, n=n)