from .local_llm import local_llm, LlamaLlmFactory
from .openai_llm import openai_llm, OpenaiLlmFactory

all_llms = [local_llm, openai_llm]
all_llm_factories = [LlamaLlmFactory(), OpenaiLlmFactory()]

__all__ = [
    "local_llm",
    "LlamaLlmFactory",
    "openai_llm",
    "OpenaiLlmFactory",
    "all_llms",
    "all_llm_factories",
]
