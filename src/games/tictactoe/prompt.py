from prompt import get_answer_format, extract_final_answer

def create_prompt(game_state):
    player_symbol_dict = {-1: "X", 0: "-", 1: 'O'}
    board = board_to_str(game_state["board"], player_symbol_dict)
    player_symbol = player_symbol_dict[game_state["next_player"]]
    n_dim = len(board)
    
    prompt=f"""
We are playing tic-tac-toe. You are {player_symbol}. It is your move. The current board is:
    
```
{board}
```

Think carefully about the optimal move and then choose the space to move in. Your final answer should be of the format "{get_answer_format("row, col")}" where row is one of {{top, bottom, center}} and col is one of {{left, center, right}}.
""".strip()

    return prompt

def response_to_move(model_response):
    row_text_to_idx = {"top": 0, "center": 1, "bottom": 2}
    col_text_to_idx = {"left": 0, "center": 1, "right": 2}
    try:
        answer = extract_final_answer(model_response)
        row, col = answer.lower().split(",")
        row, col = row.strip(), col.strip()
        return (row_text_to_idx[row], col_text_to_idx[col])
    except:
        return None
    
# def response_to_move(model_response):
#     try:
#         answer = extract_final_answer(model_response)
#         row, col = answer.lower().split(",")
#         return (int(row.strip())-1, int(col.strip())-1)
#     except:
#         return None

def board_to_str(board, player_symbol_dict):
    board_str = "\n".join([" ".join([player_symbol_dict[x] for x in row]) for row in board])
    return board_str