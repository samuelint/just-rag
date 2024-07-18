from typing import Optional
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv, find_dotenv

from just_rag.llm_factory import LLMFactory

_ = load_dotenv(find_dotenv())

openai_llm = ChatOpenAI(model="gpt-3.5-turbo")


class OpenaiLlmFactory(LLMFactory):
    @property
    def _llm_type(self) -> str:
        return "openai"

    def __call__(
        self,
        temperature: Optional[float] = 0.0,
    ):
        return ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=temperature,
        )
