from src.prompt import get_answer_format, extract_final_answer

ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
def create_prompt(game_state):
    available_words = game_state["board"][0]
    next_player_idx = 1 if game_state["next_player"] == 1 else 2
    other_player_idx = 2 if game_state["next_player"] == 1 else 1
    next_player_words = game_state["board"][next_player_idx]
    other_player_words = game_state["board"][other_player_idx]
    
    prompt=f"""
You are playing Word Connect, a game where you take turns choosing a word from a shared pool of words until one of you has three distinct words that all share a letter. The first player to get three distinct words that all share a letter wins. It is your turn. The current available words, your words, and your opponent's words are below:
   
Available Words: {available_words}
Your Words: {next_player_words}
Opponent's Words: {other_player_words}

Think carefully about the optimal move and then choose the word to take from the available words. Your final answer should be a single word of the format {get_answer_format("word")} indicating which word you are taking from the available words.
""".strip()

    return prompt

def response_to_move(model_response):
    try:
        answer = extract_final_answer(model_response).lower().strip()
        return answer
    except:
        return None