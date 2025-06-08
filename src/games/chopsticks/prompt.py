from src.prompt import get_answer_format, extract_final_answer

def create_prompt(game_state):
    player_idx = 0 if game_state["next_player"] == 1 else 1
    player_hand = game_state["board"][player_idx]
    opponent_hand = game_state["board"][1-player_idx]
    prompt=f"""
We are playing Overflow, a two player game where you and your opponent are competing to zero out each other's numbers. You each have two numbers from zero to five. Each turn you can choose from one of two types of moves:

Attack: In an attack, you choose one of your non-zero numbers and use it to attack one of your opponent's non-zero numbers. Your opponent's number becomes the sum of the two numbers mod 5. So for instance, if you attack your opponent's number B with your number A, then your opponent's number B is now (B + A) mod 5. Remember, to perform an attack, A and B must start as non-zero numbers.

Fortify: If you choose to fortify, then you take the sum of your two numbers (A + B) and restribute this sum in a new way (C + D) where A + B = C + D and both C and D are not equal to A or B. C and D also must be less than 5. For instance, if your numbers are A: 1 and B: 3, then you can fortify this into your new numbers A: 2, B: 2 or A: 0, B: 4 but you cannot fortify this into A: 3, B: 1.

The goal of the game is to change both of your opponent's numbers into zeros. The first player to do this wins.
    
Your Numbers
A: {player_hand[0]}
B: {player_hand[1]}

Opponent's Numbers
A: {opponent_hand[0]}
B: {opponent_hand[1]}

If you choose to attack, your final answer should be of the format "{get_answer_format("your_letter, opponent_letter")}" where your_letter is "A" or "B" representing the number you are attacking with and opponent_letter is also "A" or "B" representing the number you are attacking. 

If you choose to fortify, your final answer should be of the format "{get_answer_format("new_A, new_B")}" where new_A is the number you are changing A to (must be different from the current A and B and also less than 5) and new_B is the number you are changing B to (must be different from the current A and B and also less than 5). The numbers new_A and new_B must have the same sum as A and B.

Think carefully about the optimal move and then answer. 
""".strip()

    return prompt

def response_to_move(model_response):
    try:
        answer = extract_final_answer(model_response)
        x, y = answer.lower().split(",")
        x, y = x.strip(), y.strip()
        letters = ["a", "b"]
        if x in letters and y in letters:
            return ("TAP", letters.index(x), letters.index(y))
        else:
            return ("SPLIT", int(x), int(y))
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