from prompt import get_answer_format, extract_final_answer

ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
def create_prompt(game_state):
    available_nums = game_state["board"][0]
    next_player_idx = 1 if game_state["next_player"] == 1 else 2
    other_player_idx = 2 if game_state["next_player"] == 1 else 1
    next_player_nums = game_state["board"][next_player_idx]
    other_player_nums = game_state["board"][other_player_idx]
    goal_sum = game_state["goal_sum"]

    prompt=f"""
You are playing Grab Bag, a game where you and your opponent take turns choosing a number from a shared pool of numbers until one of you has three distinct numbers that add to {goal_sum}. The first player to get three distinct numbers that add to {goal_sum} wins. It is your turn. The current available numbers, your numbers, and your opponent's numbers are below:
   
Available Numbers: {available_nums}
Your Numbers: {next_player_nums}
Opponent's Numbers: {other_player_nums}

Think carefully about the optimal move and then choose the number to take from the original numbers. Your final answer should be a single number of the format {get_answer_format("number")} indicating which number you are taking from the available numbers.
""".strip()

    return prompt

def response_to_move(model_response):
    try:
        answer = extract_final_answer(model_response)
        answer = int(answer)
        return answer
    except:
        return None
    
# def response_to_move(model_response):
#     try:
#         answer = extract_final_answer(model_response)
#         row, col = answer.lower().split(",")
#         return (int(row.strip())-1, int(col.strip())-1)
#     except:
#         return None