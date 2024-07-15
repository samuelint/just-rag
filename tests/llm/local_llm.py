import os
from llama_cpp import Llama, Optional
from langchain_llamacpp_chat_model import LlamaChatModel

from rag_compare.llm_factory import LLMFactory

model_path = os.path.join(
    os.path.expanduser("~/.cache/lm-studio/models"),
    "lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf",
)

llama = Llama(
    verbose=True,
    model_path=model_path,
    n_ctx=8192,
    n_batch=224,  # Should be between 1 and n_ctx, consider the amount of RAM.
    offload_kqv=True,  # Equivalent of f16_kv=True
    n_gpu_layers=-1,  # -1 is all on GPU
    chat_format="chatml-function-calling",  # https://llama-cpp-python.readthedocs.io/en/latest/#function-calling
)
local_llm = LlamaChatModel(llama=llama, temperature=0.0)


class LlamaLlmFactory(LLMFactory):
    @property
    def _llm_type(self) -> str:
        return "llama"

    def __call__(
        self,
        temperature: Optional[float] = 0.0,
    ):
        return LlamaChatModel(
            llama=llama,
            temperature=temperature,
        )
