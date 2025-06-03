from src.prompt import get_answer_format, extract_final_answer

ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
def create_prompt(game_state):
    player_symbol_dict = {-1: "X", 0: "-", 1: 'O'}
    board = board_to_str(game_state["board"], player_symbol_dict)
    player_symbol = player_symbol_dict[game_state["next_player"]]
    opponent_symbol = player_symbol_dict[-1*game_state["next_player"]]
    nrows, ncols = len(game_state["board"]), len(game_state["board"][0])
    final_col_letter = ALPHABET[ncols-1]
    prompt=f"""
You are playing Connect Four on a board that has {nrows} rows and {ncols} columns. You are {player_symbol} and your opponent is {opponent_symbol}. A {player_symbol_dict[0]} indicates an empty spot. It is your move. The current board is:
    
```
{board}
```

Think carefully about the optimal move and then choose the column to place your piece in. Your final answer should be a single letter of the format {get_answer_format("column_letter")} indicating which column to place your piece in. The columns are lettered A through {final_col_letter} from left to right.
""".strip()

    return prompt

def response_to_move(model_response, game):
    try:
        answer = extract_final_answer(model_response)
        answer = [letter for letter in answer.upper() if letter in ALPHABET]
        assert len(answer) == 1
        assert answer in ALPHABET[:len(game.board[0])]
        return ALPHABET.index(answer)
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
    column_letters = list(ALPHABET[:len(board[0])])
    board_str = " ".join(column_letters) + "\n" + "-"*len(column_letters)*2 + "\n"
    board_str += "\n".join([" ".join([player_symbol_dict[x] for x in row]) for row in board])
    return board_str