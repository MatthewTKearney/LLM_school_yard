from inspect_ai import Task, task
from inspect_ai.scorer import Score, accuracy, scorer, stderr
from inspect_ai.solver import TaskState, generate
from inspect_ai.dataset import json_dataset, Sample
from src.prompts.tictactoe import create_prompt, evaluate_response

def record_to_sample(record):
    return Sample(
        input=create_prompt(record),
        metadata=record
    )

tictactoe = json_dataset(
    "/home/ubuntu/LLM_school_yard/data/critical_points/tic-tac-toe.json", 
    record_to_sample
)


@scorer(metrics=[accuracy(), stderr()])
def tictactoe_scorer():
    async def score(state: TaskState, target):
        score = evaluate_response(state.metadata, state.output.completion)
        if score is None:
            return Score(value="Error")
        elif score == 1:
            return Score(value="C")
        else:
            return Score(value="I")
    return score

@task
def security_guide():
    return Task(
        dataset=tictactoe,
        solver=[generate()],
        scorer=tictactoe_scorer()
    )
