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

# PROMPT_TYPES
# def record_to_sample(record, game_type, prompt_type):
#     prompt_from_record = PROMPT_TYPES[game_type][prompt_type]
#     answers_from_record = ANSWER_TYPES[game_type][prompt_type]
#     return Sample(
#         input = prompt_from_record(record),
#         target = answers_from_record(record)
#         id = 0,
#         metadata = record
#     )

# dataset = json_dataset("fpath", record_to_sample)

# dataset.filter(lambda sample: sample.metadata["x"] == "y")

# dataset = dataset.shuffle()

# def get_task(dataset):
#     return Task(
#         dataset=dataset,
#         solver=[
#             generate(),
#             answer(),
#         ],
#         scorer=score_result()
#     )