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

def get_model_name(model_name, model_config):
    keys_to_include = ["reasoning_effort", "reasoning_tokens"]
    appendix = ""
    for key in keys_to_include:
        if key in vars(model_config) and vars(model_config)[key] is not None:
            appendix += "_"+str(vars(model_config)[key])
    return f"{model_name}{appendix}"

def get_models(model_names, default_model_config={}):
    parsed_model_names = []
    generation_configs = []
    for model_name in model_names:
        matched=False
        for pattern, (name, generation_kwargs_fxn) in MODEL_TO_GENERATION_CONFIG.items():
            if re.fullmatch(pattern, model_name):
                parsed_model_names.append(name)
                generation_config = copy(default_model_config)
                generation_config.update(generation_kwargs_fxn(model_name))
                generation_configs.append(generation_config)
                matched=True
                break
        if not matched:
            parsed_model_names.append(model_name)
            generation_configs.append({})

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
    

    
