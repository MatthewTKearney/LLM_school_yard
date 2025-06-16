from strategy_utils import random_strategy, random_subset_strategy
from functools import partial
import numpy as np

def always_kill(samples, secondary_strategy=random_strategy):
    assert len(samples) > 0
    choice_sets = []
    for sample in samples:
        next_player_idx = 0 if sample["next_player"] == 1 else 1
        next_player_hand = sample["board"][next_player_idx]
        opponent_hand = sample["board"][1-next_player_idx]
        choice_set = set()
        for next_player_idx, next_player_num in enumerate(next_player_hand):
            for opponent_idx, opponent_num in enumerate(opponent_hand):
                if (next_player_num + opponent_num) == 5:
                    choice_set.add(("TAP", next_player_num, opponent_num))
        if len(choice_set)> 0:
            choice_sets.append(choice_set)
        else:
            choice_sets.append(secondary_strategy([sample])[0])
    return choice_sets

def keep_at_one(samples, secondary_strategy=random_strategy):
    assert len(samples) > 0
    choice_sets = []
    for sample in samples:
        next_player_idx = 0 if sample["next_player"] == 1 else 1
        next_player_hand = sample["board"][next_player_idx]
        opponent_hand = sample["board"][1-next_player_idx]
        choice_set = set()
        if sorted(opponent_hand) == [0,1]:
            hand_sum = sum(next_player_hand)
            legal_splits = np.array([[i, hand_sum-i] for i in range(hand_sum+1)])
            if len(legal_splits) > 0:
                legal_splits = legal_splits[np.all(legal_splits >= 0, axis=1) & np.all(legal_splits < 5, axis=1) & np.all(legal_splits != next_player_hand[0], axis=1)]
            if len(legal_splits) > 0:
                choice_set = set(("SPLIT", x, y) for x, y in legal_splits.to_list())
        if len(choice_set) == 0:
            choice_set = secondary_strategy([sample])[0]
        choice_sets.append(choice_set)
    return choice_sets

def get_strategies():
    strategies = {
        "always_kill": always_kill,
        "keep_at_one": keep_at_one,
        "keep_at_one_and_kill": partial(keep_at_one, secondary_strategy=always_kill),
    }
    return strategies