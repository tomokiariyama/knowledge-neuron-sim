from transformers import (
    GPTNeoXForCausalLM,
    AutoModelForCausalLM,
    AutoTokenizer,
)
from .knowledge_neurons import KnowledgeNeurons
from .data import pararel, pararel_expanded, PARAREL_RELATION_NAMES

PYTHIA_MODELS = [
    "EleutherAI/pythia-70m-deduped",
    "EleutherAI/pythia-160m-deduped",
    "EleutherAI/pythia-410m-deduped",
    "EleutherAI/pythia-1b-deduped",
    "EleutherAI/pythia-1.4b-deduped",
    "EleutherAI/pythia-2.8b-deduped",
    "EleutherAI/pythia-6.9b-deduped",
    "EleutherAI/pythia-12b-deduped",
]
OLMO_MODELS = [
    "allenai/OLMo-1B-0724-hf",
    "allenai/OLMo-2-1124-7B",
]
ALL_MODELS = PYTHIA_MODELS + OLMO_MODELS


def initialize_model_and_tokenizer(model_name: str, step: str):
    if model_name in PYTHIA_MODELS:
        tokenizer = AutoTokenizer.from_pretrained(model_name, revision=f"step{step}")
        model = GPTNeoXForCausalLM.from_pretrained(model_name, revision=f"step{step}")
    elif model_name in OLMO_MODELS:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
    else:
        raise ValueError(f"Model {model_name} not supported")

    model.eval()

    return model, tokenizer


def model_type(model_name: str):
    if model_name in PYTHIA_MODELS:
        return "gpt_neox"
    elif model_name in OLMO_MODELS:
        return "olmo"
    else:
        raise ValueError(f"Model {model_name} not supported")
