import argparse 

from src.game_tree import export_game_states
from src.eval_task import run_task
from src.process_results import process_results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--game", help="Game to generate tree from", required=True, type=str)
    parser.add_argument("--data_dir", help="Directory to where critical points are saved", default="./data", type=str)
    
    # args for exporting game states
    parser.add_argument("--include_all_states", help="Whether to export only the critical points in the game tree or all states", action='store_true')
    
    # args for running task
    parser.add_argument("--sample_filters", help="Name of filters to use to filter samples before getting model responses", default=None, nargs="+")
    parser.add_argument("--models", help="Models to evaluate", required=True, nargs='+')
    parser.add_argument("--max_samples", help="Maximum number of game states to evaluate model on", default=None, type=int)
    parser.add_argument("--max_samples_per_difficulty", help="Maximum number of game states of each difficulty to evaluate model on", default=None, type=int)
    parser.add_argument("--token_limit", help="Model generation token limit", default=None, type=int)
    parser.add_argument("--reasoning_tokens", help="Model generation token limit", default=None, type=int)
    parser.add_argument("--reasoning_effort", help="Model generation token limit", default=None, type=str)
    parser.add_argument("--reasoning_summary", help="Model generation token limit", action="store_true")

    # args for processing results
    parser.add_argument("--baselines", help="Baseline strategies to plot evaluation results for", required=False, default="all", nargs='+')
    parser.add_argument("--group_keys", help="Properties of samples for grouping scores into subplots", default=None, nargs='+')

    args = parser.parse_args()

    export_game_states(args.game, outdir=args.data_dir, include_all_states=False)

    print(args.token_limit)
    run_task(args.game, args.models, data_dir=args.data_dir, sample_filters=args.sample_filters, max_samples=args.max_samples, max_samples_per_difficulty=args.max_samples_per_difficuly, token_limit=args.token_limit, reasoning_tokens=args.reasoning_tokens, reasoning_effort=args.reasoning_effort, reasoning_summary=args.reasoning_summary)

    process_results(args.game, args.models, args.baselines, args.group_keys, args.data_dir)

if __name__ == "__main__":
    main()