"""
Unified LLM client for the environment.

Three backends:
  - OpenAI-compatible (vLLM/local) — for self-hosted models (default)
  - HF Inference API — free, serverless, no infra needed
  - Anthropic (Claude) — external judge, frees GPU memory
"""

import os
import json
import logging
import re
import time

logger = logging.getLogger(__name__)


class LLMClient:
    """
    Thin wrapper that picks the right backend based on env vars.

    Config:
      LLM_BACKEND=openai     (default) → uses OpenAI-compatible endpoint (vLLM on H100)
      LLM_BACKEND=hf                   → uses HF Inference API (requires credits)
      LLM_BACKEND=anthropic            → uses Anthropic Claude API

    OpenAI mode env vars:
      LLM_BASE_URL  — vLLM endpoint (default: http://localhost:8001/v1)
      LLM_API_KEY   — API key (default: "local")
      LLM_MODEL     — model name

    HF mode env vars:
      HF_TOKEN      — HuggingFace token
      LLM_MODEL     — model ID (default: Qwen/Qwen3-14B)

    Anthropic mode env vars:
      ANTHROPIC_API_KEY — Anthropic API key
      LLM_MODEL         — model name (default: claude-sonnet-4-20250514)
    """

    def __init__(self):
        self.backend = os.environ.get("LLM_BACKEND", "openai")
        default_model = "claude-sonnet-4-20250514" if self.backend == "anthropic" else "Qwen/Qwen3-14B"
        self.model = os.environ.get("LLM_MODEL", default_model)

        if self.backend == "anthropic":
            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError(
                    "ANTHROPIC_API_KEY env var required when LLM_BACKEND=anthropic. "
                    "Set it via: export ANTHROPIC_API_KEY=sk-ant-..."
                )
            from anthropic import Anthropic
            self.client = Anthropic(api_key=api_key)
            logger.info(f"LLM backend: Anthropic ({self.model})")
        elif self.backend == "hf":
            from huggingface_hub import InferenceClient
            self.client = InferenceClient(
                model=self.model,
                token=os.environ.get("HF_TOKEN"),
            )
            logger.info(f"LLM backend: HF Inference API ({self.model})")
        else:
            from openai import OpenAI
            self.client = OpenAI(
                base_url=os.environ.get("LLM_BASE_URL", "http://localhost:8001/v1"),
                api_key=os.environ.get("LLM_API_KEY", "local"),
            )
            logger.info(f"LLM backend: OpenAI-compatible ({self.model})")

    def chat(self, system: str, user: str, temperature: float = 0.3, max_tokens: int = 1024) -> str:
        """Send a chat completion request. Returns the raw response text."""
        if self.backend == "anthropic":
            return self._chat_anthropic(system, user, temperature, max_tokens)
        elif self.backend == "hf":
            return self._chat_hf(system, user, temperature, max_tokens)
        return self._chat_openai(system, user, temperature, max_tokens)

    def chat_json(self, system: str, user: str, temperature: float = 0.3, max_tokens: int = 1024) -> dict:
        """Send a chat request and parse the response as JSON."""
        raw = self.chat(system, user, temperature, max_tokens)
        return self._parse_json(raw)

    @staticmethod
    def _parse_json(raw: str) -> dict:
        """Extract and parse JSON from LLM response, handling markdown fences."""
        raw = raw.strip()
        # Strip ```json ... ``` or ``` ... ``` wrappers
        fence_match = re.search(r'```(?:json)?\s*\n?(.*?)```', raw, re.DOTALL)
        if fence_match:
            raw = fence_match.group(1).strip()
        return json.loads(raw)

    def _chat_anthropic(self, system: str, user: str, temperature: float, max_tokens: int) -> str:
        """Call Anthropic API with retry on transient errors."""
        from anthropic import APIStatusError, RateLimitError

        for attempt in range(3):
            try:
                response = self.client.messages.create(
                    model=self.model,
                    system=system,
                    messages=[{"role": "user", "content": user}],
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                return response.content[0].text
            except RateLimitError:
                wait = 2 ** attempt
                logger.warning(f"Anthropic rate limited, retrying in {wait}s...")
                time.sleep(wait)
            except APIStatusError as e:
                if e.status_code >= 500 and attempt < 2:
                    wait = 2 ** attempt
                    logger.warning(f"Anthropic server error ({e.status_code}), retrying in {wait}s...")
                    time.sleep(wait)
                else:
                    raise
        raise RuntimeError("Anthropic API failed after 3 retries")

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
