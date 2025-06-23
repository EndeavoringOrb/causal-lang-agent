from typing import List
import requests
import json

class LlamaServerClient:
    def __init__(
        self,
        base_url: str = "http://localhost:8080",
    ) -> None:
        """
        Initialize the LlamaCPPServerClient to interface with a llama.cpp server.

        :param base_url: The base URL of the llama.cpp server API (default is http://localhost:8080).
        :param model: A model identifier (optional, kept for consistency with other clients,
                      llama.cpp server typically doesn't use this directly for /v1/completions).
        """
        self.base_url = base_url.rstrip("/")

    def generate(
        self,
        prompt: str | list[dict[str, str]],
        stream: bool = True,
        stop: List[str] = [],
        log_response: bool = False,
        text_only: bool = True,
        **kwargs,
    ):
        """
        Internal method to interact with the llama.cpp server's /v1/completions endpoint.
        This method is adapted from the LlamaServerClient example provided.

        :param prompt: The input prompt string.
        :param stream: Whether to stream the response (currently collects full response if True).
        :param stop: A list of stop sequences.
        :param log_response: If stream is true, prints responses as they are streamed back.
        :param text_only: If true, only return text (assumed for this client).
        :param max_tokens: Maximum number of tokens to generate.
        :param temperature: Sampling temperature for controlling randomness.
        :param kwargs: Additional parameters to pass to the API.
        :return: The generated text as a string.
        """
        payload = {
            "stream": stream,
            **kwargs,
        }

        if isinstance(prompt, str):
            url = f"{self.base_url}/v1/completions"
            payload["prompt"] = prompt
            is_chat = False
        else:
            url = f"{self.base_url}/v1/chat/completions"
            payload["messages"] = prompt
            is_chat = True
        if stop:
            payload["stop"] = stop

        response = requests.post(url, json=payload, stream=stream)
        response.raise_for_status()

        if stream:
            for line in response.iter_lines(decode_unicode=True):
                if line:
                    if line == "data: [DONE]":
                        break
                    try:
                        line_data = json.loads(line[5:])  # Strip "data: " prefix
                        if text_only:
                            if is_chat:
                                if "content" not in line_data["choices"][0]["delta"]:
                                    continue
                                content = line_data["choices"][0]["delta"]["content"]
                                if content is None:
                                    continue
                            else:
                                content = line_data["choices"][0]["text"]
                        else:
                            content = (
                                line_data  # Handle non-text_only structure if needed
                            )
                        if log_response:
                            print(content, end="", flush=True)
                        yield content
                    except json.JSONDecodeError as e:
                        print(f"JSON decode error: {e} - Line: {line}")
                        continue
        else:
            return response.json()["choices"][0]["text"]
