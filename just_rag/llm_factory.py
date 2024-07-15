from abc import ABC, abstractmethod
from typing import Optional
from langchain_core.language_models import BaseChatModel


class LLMFactory(ABC):
    @abstractmethod
    def __call__(
        self,
        temperature: Optional[float] = None,
    ) -> BaseChatModel:
        pass
