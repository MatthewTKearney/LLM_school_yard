from games import GAME_PACKAGES

def get_difficulty(sample):
    if (not sample.metadata["win_difficulty"] is None) and sample.metadata["lose_difficulty"] is None:
        return sample.metadata["win_difficulty"]
    elif (not sample.metadata["lose_difficulty"] is None) and sample.metadata["win_difficulty"] is None:
        return sample.metadata["lose_difficulty"]
    else:
        raise ValueError

def get_filter(filter_name, game=None):
    if filter_name.startswith("difficulty_gt"):
        diff = int(filter_name.split("_")[-1])
        return lambda sample: get_difficulty(sample) > diff
    elif filter_name.startswith("difficulty_lt"):
        diff = int(filter_name.split("_")[-1])
        return lambda sample: get_difficulty(sample) < diff
    elif filter_name.startswith("difficulty_range"):
        lower, upper = list(filter(lambda x: int(x), filter_name.split("_")[-2:]))
        return lambda sample: get_difficulty(sample) >= lower and get_difficulty(sample) <= upper
    elif filter_name.startswith("difficulty"):
        diff = int(filter_name.split("_")[-1])
        return lambda sample: get_difficulty(sample) == diff
    else:
        return GAME_PACKAGES[game].critical_point_filters[filter_name]