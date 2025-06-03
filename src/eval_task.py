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

from src.games import GAME_PACKAGES
from src.prompt import score_response

def record_to_sample(record, create_prompt):
    record = json_loads(json_dumps(record)) #for custom encoder
    return Sample(
        input=create_prompt(record),
        metadata=record
    )

def get_dataset(game, data_dir="./data", sample_filter=None, max_num_samples=None, seed=None):
    dataset_path = os.path.join(data_dir, f"critical_points/{game}.json")
    game_package = GAME_PACKAGES[game]
    dataset = json_dataset(
        dataset_path, 
        partial(record_to_sample, create_prompt=game_package.prompt.create_prompt),
        shuffle=True,
        seed=seed,
    )

    # always filter for points that are either win_critical or lose_critical but not both
    dataset = dataset.filter(lambda sample: sample.metadata["win_critical"] != sample.metadata["lose_critical"])
    if sample_filter is not None:
        dataset = dataset.filter(game_package.critical_points.filters[sample_filter])
    if max_num_samples:
        dataset = dataset[:max_num_samples]
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

def run_task(game, models, data_dir="./data", sample_filter=None, max_num_samples=None, random_seed=0, token_limit=None):
    if random_seed:
        random.seed(random_seed)
    dataset = get_dataset(game, data_dir=data_dir, sample_filter=sample_filter, max_num_samples=max_num_samples, seed=random_seed)
    scorer = get_scorer(game)
    task = get_game_task(dataset, scorer)
    log_dir = os.path.join(data_dir, "model_evals", game)
    logs = eval(
            task,
            model=models,
            log_dir=log_dir,
            token_limit=token_limit, 
            max_connections=20,  
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
    parser.add_argument("--sample_filter", help="Name of the game specific function to use to filter samples before getting model responses", default=None, type=str)
    parser.add_argument("--max_num_samples", help="Maximum number of game states to evaluate model on", default=None, type=int)
    parser.add_argument("--random_seed", help="Random seed for shuffling data", default=None, type=int)
    parser.add_argument("--token_limit", help="Model generation token limit", default=None, type=int)
    args = parser.parse_args()
    run_task(args.game, args.models, args.data_dir, args.sample_filter, args.max_num_samples, args.random_seed)

if __name__ == "__main__":
    main()

