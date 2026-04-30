"""
LLM service for chat completions using Ollama.

This service handles LLM chat operations with:
- Structured JSON output support with fallback parsing
- Configurable temperature and model selection
- Graceful error handling
"""

from typing import Dict, List, Optional, Any
import json
import re

import requests

from services.config.settings import OllamaConfig


class LLMService:
    """Handles LLM chat completions with structured output support."""

    def __init__(self, config: OllamaConfig):
        """
        Initialize LLM service.

        Args:
            config: Ollama configuration with host, model, timeouts.
        """
        self.config = config
        self.is_openrouter = bool(config.openrouter_api_key)
        if self.is_openrouter:
            self._url = "https://openrouter.ai/api/v1/chat/completions"
        else:
            self._url = f"{config.host}/api/chat"

    def chat(
        self,
        prompt: str,
        system_prompt: str = "",
        temperature: float = 0.0,
        model: Optional[str] = None,
        top_p: float = 0.9,
        require_json: bool = False,
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        Send chat request and return response text.

        Args:
            prompt: User prompt/question.
            system_prompt: Optional system prompt for context.
            temperature: Sampling temperature (0.0 = deterministic).
            model: Optional model override.
            top_p: Top-p sampling parameter.
            max_tokens: Optional maximum tokens in the response.

        Returns:
            Response text from LLM, or error message string.
        """
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": model or self.config.chat_model,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
        }

        if not self.is_openrouter:
            payload["stream"] = False
            if require_json:
                payload["format"] = "json"
            if max_tokens:
                payload["options"] = {
                    "temperature": temperature,
                    "top_p": top_p,
                    "num_predict": max_tokens,
                }
            else:
                payload["options"] = {
                    "temperature": temperature,
                    "top_p": top_p,
                }
            del payload["temperature"]
            del payload["top_p"]
        else:
            if require_json:
                payload["response_format"] = {"type": "json_object"}
            if max_tokens:
                payload["max_tokens"] = max_tokens

        headers = {}
        if self.is_openrouter:
            headers = {
                "Authorization": f"Bearer {self.config.openrouter_api_key}",
                "HTTP-Referer": "http://localhost:3000",
                "X-Title": "MACKIS RAG Eval"
            }

        try:
            response = requests.post(
                self._url,
                json=payload,
                headers=headers,
                timeout=self.config.chat_timeout
            )

            if response.status_code != 200:
                provider = "OpenRouter" if self.is_openrouter else "Ollama"
                return f"{provider} Error: {response.text}"

            data = response.json()
            if self.is_openrouter:
                content = data.get("choices", [{}])[0].get("message", {}).get("content")
                if content is None:
                    content = ""
            else:
                content = data.get("message", {}).get("content", "")

            if not content:
                return "Empty response from LLM."

            return content

        except requests.exceptions.Timeout:
            return "Connection timeout: LLM request took too long."
        except requests.exceptions.ConnectionError:
            return "Connection error: Could not connect to Ollama."
        except Exception as e:
            return f"LLM error: {e}"

    def chat_json(
        self,
        prompt: str,
        system_prompt: str = "",
        temperature: float = 0.0,
        model: Optional[str] = None,
    ) -> Optional[Dict]:
        """
        Chat expecting JSON response, with fallback parsing.

        Args:
            prompt: User prompt/question.
            system_prompt: System prompt (should instruct JSON output).
            temperature: Sampling temperature.
            model: Optional model override.

        Returns:
            Parsed JSON dict, or None if parsing fails.
        """
        text = self.chat(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=temperature,
            model=model,
            require_json=True
        )

        if not text:
            return None

        # Check for error responses
        if text.startswith(("Ollama Error", "Connection", "LLM error", "Empty")):
            print(f"[LLMService] {text}")
            return None

        # Try direct JSON parse
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try extracting JSON from response (LLM might include extra text)
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass

        # Try extracting JSON array
        match = re.search(r"\[.*\]", text, flags=re.DOTALL)
        if match:
            try:
                return {"items": json.loads(match.group(0))}
            except json.JSONDecodeError:
                pass

        print(f"[LLMService] Failed to parse JSON from: {text[:200]}...")
        return None

    def chat_with_history(
        self,
        prompt: str,
        history: List[Dict[str, str]],
        system_prompt: str = "",
        temperature: float = 0.0,
        model: Optional[str] = None,
    ) -> str:
        """
        Chat with conversation history.

        Args:
            prompt: Current user prompt.
            history: List of previous messages with 'role' and 'content'.
            system_prompt: Optional system prompt.
            temperature: Sampling temperature.
            model: Optional model override.

        Returns:
            Response text from LLM.
        """
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        # Add history
        for msg in history:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if content:
                messages.append({"role": role, "content": content})

        # Add current prompt
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": model or self.config.chat_model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "top_p": 0.9,
            },
        }

        try:
            response = requests.post(
                self._url,
                json=payload,
                timeout=self.config.chat_timeout
            )

            if response.status_code != 200:
                return f"Ollama Error: {response.text}"

            data = response.json()
            return data.get("message", {}).get("content", "Empty response.")

        except Exception as e:
            return f"LLM error: {e}"

    def is_available(self) -> bool:
        """
        Check if Ollama service is available.

        Returns:
            True if service is reachable.
        """
        try:
            response = requests.get(
                f"{self.config.host}/api/tags",
                timeout=5
            )
            return response.status_code == 200
        except Exception:
            return False

    def list_models(self) -> List[str]:
        """
        List available models.

        Returns:
            List of model names, or empty list on error.
        """
        try:
            response = requests.get(
                f"{self.config.host}/api/tags",
                timeout=10
            )
            if response.status_code == 200:
                data = response.json()
                return [m.get("name", "") for m in data.get("models", [])]
        except Exception:
            pass
        return []
