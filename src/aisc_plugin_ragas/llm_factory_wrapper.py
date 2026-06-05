"""Custom LangChain BaseChatModel that routes generation through LLM Factory."""
from __future__ import annotations

from typing import Any, Iterator, List, Optional

import requests
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult


class LLMFactoryChat(BaseChatModel):
    """LangChain-compatible chat model backed by the LLM Factory /execute_prompt endpoint."""

    llm_factory_url: str
    model_name: str
    temperature: float = 0.2
    max_tokens: int = 512

    @property
    def _llm_type(self) -> str:
        return "llm-factory"

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        # Concatenate messages into a single prompt string
        prompt = "\n".join(m.content for m in messages if hasattr(m, "content"))
        payload = {
            "model": self.model_name,
            "prompt": prompt,
        }
        response = requests.post(
            f"{self.llm_factory_url}/execute_prompt",
            json=payload,
            timeout=300,
        )
        response.raise_for_status()
        text = response.json().get("response", "")
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=text))])

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGeneration]:
        result = self._generate(messages, stop=stop, run_manager=run_manager, **kwargs)
        yield from result.generations
