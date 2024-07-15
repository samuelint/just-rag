from .local_llm import local_llm
from .openai_llm import openai_llm

all_llms = [local_llm, openai_llm]

__all__ = [
    "local_llm",
    "openai_llm",
    "all_llms",
]
