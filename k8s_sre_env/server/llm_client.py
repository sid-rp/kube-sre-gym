"""
Unified LLM client for the environment.

Two backends:
  - HF Inference API (default) — free, serverless, no infra needed
  - OpenAI-compatible (vLLM/local) — for self-hosted models

The environment doesn't care which backend is used.
"""

import os
import json
import logging

logger = logging.getLogger(__name__)


class LLMClient:
    """
    Thin wrapper that picks the right backend based on env vars.

    Config:
      LLM_BACKEND=hf (default) → uses HF Inference API
      LLM_BACKEND=openai       → uses OpenAI-compatible endpoint (vLLM, local)

    HF mode env vars:
      HF_TOKEN      — HuggingFace token
      LLM_MODEL     — model ID (default: Qwen/Qwen2.5-72B-Instruct)

    OpenAI mode env vars:
      LLM_BASE_URL  — endpoint (default: http://localhost:8000/v1)
      LLM_API_KEY   — API key (default: "local")
      LLM_MODEL     — model name
    """

    def __init__(self):
        self.backend = os.environ.get("LLM_BACKEND", "hf")
        self.model = os.environ.get("LLM_MODEL", "Qwen/Qwen2.5-72B-Instruct")

        if self.backend == "hf":
            from huggingface_hub import InferenceClient
            self.client = InferenceClient(
                model=self.model,
                token=os.environ.get("HF_TOKEN"),
            )
            logger.info(f"LLM backend: HF Inference API ({self.model})")
        else:
            from openai import OpenAI
            self.client = OpenAI(
                base_url=os.environ.get("LLM_BASE_URL", "http://localhost:8000/v1"),
                api_key=os.environ.get("LLM_API_KEY", "local"),
            )
            logger.info(f"LLM backend: OpenAI-compatible ({self.model})")

    def chat(self, system: str, user: str, temperature: float = 0.3, max_tokens: int = 300) -> str:
        """Send a chat completion request. Returns the raw response text."""
        if self.backend == "hf":
            return self._chat_hf(system, user, temperature, max_tokens)
        return self._chat_openai(system, user, temperature, max_tokens)

    def chat_json(self, system: str, user: str, temperature: float = 0.3) -> dict:
        """Send a chat request and parse the response as JSON."""
        raw = self.chat(system, user, temperature)
        # Strip markdown code fences if present
        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0]
        return json.loads(raw)

    def _chat_hf(self, system: str, user: str, temperature: float, max_tokens: int) -> str:
        response = self.client.chat_completion(
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content

    def _chat_openai(self, system: str, user: str, temperature: float, max_tokens: int) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content
