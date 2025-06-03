from inspect_ai import Task, task, eval, eval_async
from inspect_ai.scorer import Score, accuracy, scorer, stderr
from inspect_ai.solver import TaskState, generate
from inspect_ai.dataset import json_dataset, Sample

from inspect_ai.dataset._util import data_to_samples, record_to_sample_fn
from inspect_ai.dataset._sources.util import resolve_sample_files
from inspect_ai.dataset._dataset import MemoryDataset

from functools import partial
import argparse
import random 
import os
from src.utils import json_load, json_loads, json_dumps
import asyncio
import shutil
import numpy as np

from src.games import GAME_PACKAGES
from src.prompt import score_response
from src.critical_point_filters import get_filter, get_difficulty

def record_to_sample(record, create_prompt):
    record = json_loads(json_dumps(record)) #for custom encoder
    return Sample(
        input=create_prompt(record),
        metadata=record
    )

def filter_dataset(dataset, sample_filters=None, max_samples = None, max_samples_per_difficulty=None, remove_symmetries=False):
    # always filter for points that are either win_critical or lose_critical but not both
    dataset = dataset.filter(lambda sample: sample.metadata["win_critical"] != sample.metadata["lose_critical"])
    
    if sample_filters is not None:
        filter_fxn = lambda x: all([get_filter(filter_name)(x) for filter_name in sample_filters])
        dataset = dataset.filter(filter_fxn)

    if max_samples_per_difficulty is not None:
        sample_difficulties = np.array([get_difficulty(sample) for sample in dataset])
        idxs_to_include = []
        for difficulty in np.unique(sample_difficulties):
            sample_idxs = np.arange(len(dataset))[sample_difficulties == difficulty]
            print(difficulty, len(sample_idxs))
            idxs_to_include += sample_idxs[:max_samples_per_difficulty].tolist()
        idxs_to_include.sort()
        dataset = MemoryDataset(
            name=dataset.name,
            location=dataset.location,
            samples=[dataset[idx] for idx in idxs_to_include],
            shuffled=dataset.shuffled,
        )
    if max_samples:
        dataset = dataset[:max_samples]

    return dataset

def get_dataset(game, data_dir="./data", sample_filters=None, max_samples=None, max_samples_per_difficulty=None, seed=None):
    dataset_path = os.path.join(data_dir, f"critical_points/{game}.json")
    game_package = GAME_PACKAGES[game]
    dataset = json_dataset(
        dataset_path, 
        partial(record_to_sample, create_prompt=game_package.prompt.create_prompt),
        shuffle=True,
        seed=seed,
    )

    print(len(dataset))

    dataset = filter_dataset(dataset, sample_filters, max_samples, max_samples_per_difficulty)
    return dataset

def get_scorer(game):
    @scorer(metrics=[accuracy(), stderr()])
    def game_scorer():
        async def score(state: TaskState, target):
            score = score_response(
                state.output.completion,
                {tuple(move["move"]) for move in state.metadata["moves"]}, 
                state.metadata["optimal_moves"], 
                response_to_move=GAME_PACKAGES[game].prompt.response_to_move
            )
            if score is None:
                return Score(value="N") # No answer
            elif score == 1:
                return Score(value="C")
            else:
                return Score(value="I")
        return score
    return game_scorer

def get_game_task(dataset, scorer):
    @task
    def game_task():
        return Task(
            dataset=dataset,
            solver=[generate()],
            scorer=scorer()
        )
    return game_task

def run_task(game, models, data_dir="./data", sample_filters=None, max_samples=None, max_samples_per_difficulty=None, random_seed=0, token_limit=None, reasoning_tokens=None, reasoning_effort=None, reasoning_summary=None):
    if random_seed:
        random.seed(random_seed)
    dataset = get_dataset(game, data_dir=data_dir, sample_filters=sample_filters, max_samples=max_samples, max_samples_per_difficulty=max_samples_per_difficulty, seed=random_seed)
    scorer = get_scorer(game)
    task = get_game_task(dataset, scorer)
    log_dir = os.path.join(data_dir, "model_evals", game)
    logs = eval(
            task,
            model=models,
            log_dir=log_dir,
            token_limit=token_limit, 
            max_connections=20,  
            reasoning_tokens=reasoning_tokens,
            reasoning_effort=reasoning_effort,
            reasoning_summary=reasoning_summary,
    )
    model_to_log_path = {log.eval.model: log.location for log in logs}
    for model, original_log_path in model_to_log_path.items():
        model_log_dir = os.path.join(log_dir, model)
        os.makedirs(model_log_dir, exist_ok=True)
        shutil.move(original_log_path, os.path.join(model_log_dir, os.path.basename(original_log_path)))
    return logs
    
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", help="Game to generate tree from", required=True, type=str)
    parser.add_argument("--models", help="Models to evaluate", required=True, nargs='+')
    parser.add_argument("--data_dir", help="Directory to where critical points are saved", default="./data", type=str)
    parser.add_argument("--sample_filters", help="Name of the filters to use to filter samples before getting model responses", default=None, nargs='+')
    parser.add_argument("--max_samples", help="Maximum number of game states to evaluate model on", default=None, type=int)
    parser.add_argument("--max_samples_per_difficulty", help="Maximum number of game states of each difficulty to evaluate model on", default=None, type=int)
    parser.add_argument("--random_seed", help="Random seed for shuffling data", default=None, type=int)
    parser.add_argument("--token_limit", help="Model generation token limit", default=None, type=int)
    parser.add_argument("--reasoning_tokens", help="Model generation token limit", default=None, type=int)
    parser.add_argument("--reasoning_effort", help="Model generation token limit", default=None, type=str)
    parser.add_argument("--reasoning_summary", help="Model generation token limit", action="store_true")
    args = parser.parse_args()
    run_task(args.game, args.models, data_dir=args.data_dir, sample_filters=args.sample_filters, max_samples=args.max_samples, max_samples_per_difficulty=args.max_samples_per_difficulty, random_seed=args.random_seed, reasoning_tokens=args.reasoning_tokens, reasoning_effort=args.reasoning_effort, reasoning_summary=args.reasoning_summary)

if __name__ == "__main__":
    main()

