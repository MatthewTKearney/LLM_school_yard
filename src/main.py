from inspect_ai import Task, task
from inspect_ai.scorer import Score, accuracy, scorer, stderr
from inspect_ai.solver import TaskState, generate
from inspect_ai.dataset import json_dataset, Sample
from prompts.tictactoe import create_prompt, evaluate_response
import random 

random.seed(0)

def record_to_sample(record):
    return Sample(
        input=create_prompt(record),
        metadata=record
    )

tictactoe = json_dataset(
    "/home/ubuntu/LLM_school_yard/data/critical_points/tictactoe.json", 
    record_to_sample,
    shuffle=True,
)

def non_trivial(sample) -> bool:
    cond_1 = sample.metadata["win_difficulty"] != 1 and sample.metadata["lose_difficulty"] != 2
    # move_to_outcome = {tuple(m["move"]): m["outcome"] for m in sample.metadata["moves"]}
    # cond_2 = (1,1) in move_to_outcome and move_to_outcome[(1, 1)] == max(move_to_outcome.values())
    return cond_1 # and not cond_2

# def win_diff_1(sample) -> bool:
#     return sample.metadata["win_difficulty"] == 1

# def lose_diff_2(sample) -> bool:
#     return sample.metadata["lose_difficulty"] == 2

tictactoe = tictactoe.filter(non_trivial)
print(len(tictactoe))

@scorer(metrics=[accuracy(), stderr()])
def tictactoe_scorer():
    async def score(state: TaskState, target):
        score = evaluate_response(state.metadata, state.output.completion)
        if score is None:
            return Score(value="N") # No answer
        elif score == 1:
            return Score(value="C")
        else:
            return Score(value="I")
    return score

@task
def tictactoe_task():
    return Task(
        dataset=tictactoe,
        solver=[generate()],
        scorer=tictactoe_scorer()
    )

