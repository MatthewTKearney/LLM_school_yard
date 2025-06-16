from collections import defaultdict
import re
from copy import copy

from inspect_ai.model import get_model, GenerateConfig

MODEL_TO_GENERATION_CONFIG = {}

REASONING_EFFORT_MODELS = [
    "openai/o3",
    "openai/o4-mini",
]
REASONING_EFFORTS = "|".join(["low", "medium", "high"])
MODEL_TO_GENERATION_CONFIG.update({
    f"{model}_({REASONING_EFFORTS})": (model, lambda x: {"reasoning_effort": x.split("_")[-1]})
      for model in REASONING_EFFORT_MODELS
})

REASONING_TOKEN_MODELS = [
    "anthropic/claude-opus-4-0",
    "anthropic/claude-sonnet-4-0",
    "anthropic/claude-3-7-sonnet-latest",
    "anthropic/claude-opus-4-20250514",
    "anthropic/claude-sonnet-4-20250514",
    "anthropic/claude-3-7-sonnet-20250219",
]
MODEL_TO_GENERATION_CONFIG.update({
    f"{model}_[0-9]+": (model, lambda x: {"reasoning_tokens": int(x.split("_")[-1].strip())})
      for model in REASONING_TOKEN_MODELS
})

def get_model_name_and_config(model_name, default_model_config):
    for pattern, (name, generation_kwargs_fxn) in MODEL_TO_GENERATION_CONFIG.items():
        if re.fullmatch(pattern, model_name):
            generation_config = copy(default_model_config)
            generation_config.update(generation_kwargs_fxn(model_name))
            return name, generation_config
    return model_name, default_model_config

def get_models(model_names, default_model_config={}):
    parsed_model_names, generation_configs = zip(*[get_model_name_and_config(model_name) for model_name in model_names])
    models = [get_model(model, config=GenerateConfig(**kwargs)) for model, kwargs in zip(parsed_model_names, generation_configs)]
    return models


ALLOWED_MODEL_KWARGS = [
    "max_retries",
    "timeout",
    "max_connections",
    "system_message",
    "max_tokens",
    "top_p",
    "temperature",
    "stop_seqs",
    "best_of",
    "frequency_penalty",
    "presence_penalty",
    "logit_bias",
    "seed",
    "top_k",
    "num_choices",
    "logprobs",
    "top_logprobs",
    "parallel_tool_calls",
    "internal_tools",
    "max_tool_output",
    "cache_prompt",
    "reasoning_effort",
    "reasoning_tokens",
    "reasoning_summary",
    "reasoning_history",
    "response_schema",
    "extra_body",
]

def parse_model_config_kwargs(kwargs):
    model_config_kwargs = {
        k: v for k,v in kwargs.items() if k in ALLOWED_MODEL_KWARGS
    }
    return model_config_kwargs
    

    
