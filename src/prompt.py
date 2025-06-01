import numpy as np
def get_answer_format(answer_text):
    return f"<final_answer>{answer_text}</final_answer>"

def extract_final_answer(response):
    start_tag = "<final_answer>"
    end_tag = "</final_answer>"
    try:
        start_tag_idx = response.index(start_tag)
        end_tag_idx = response.index(end_tag)
        answer = response[start_tag_idx + len(start_tag):end_tag_idx]
        return answer
    except:
        return None
    
def score_response(model_response, legal_moves, optimal_moves, response_to_move=None, improper_response_value=None):
    if response_to_move:
        move = response_to_move(model_response)
    else:
        move = model_response
    if not move in legal_moves:
        return improper_response_value
    move_score = 1 if move in optimal_moves else 0
    return move_score