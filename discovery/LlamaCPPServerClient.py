import requests
import json
import time
from typing import List


class LlamaCPPServerClient:
    def __init__(self, base_url: str = "http://localhost:55551", model: str = "") -> None:
        """
        Initialize the LlamaCPPServerClient to interface with a llama.cpp server.

        :param base_url: The base URL of the llama.cpp server API (default is http://localhost:55551).
        :param model: A model identifier (optional, kept for consistency with other clients,
                      llama.cpp server typically doesn't use this directly for /v1/completions).
        """
        self.base_url = base_url.rstrip("/")
        self.model = model  # Stored for consistency, not directly used by the server's completions endpoint.

    def _generate(
        self,
        prompt: str,
        stream: bool = False,
        stop: List[str] = [],
        log_response: bool = True,
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
        url = f"{self.base_url}/v1/completions"
        payload = {
            "prompt": prompt,
            "stream": stream,
            "stop": stop,
            **kwargs,
        }

        response = requests.post(url, json=payload, stream=stream)
        response.raise_for_status()

        if stream:
            # For inquire_LLMs, we need a single string output.
            # Collect streamed content and return it as a complete string.
            full_response = ""
            for line in response.iter_lines(decode_unicode=True):
                if line:
                    if line == "data: [DONE]":
                        break
                    try:
                        line_data = json.loads(line[5:]) # Strip "data: " prefix
                        if text_only:
                            content = line_data["choices"][0]["text"]
                        else:
                            content = line_data # Handle non-text_only structure if needed
                        if log_response:
                            print(content, end="", flush=True)
                        full_response += content
                    except json.JSONDecodeError as e:
                        print(f"JSON decode error: {e} - Line: {line}")
                        continue
            return full_response
        else:
            return response.json()["choices"][0]["text"]

    def inquire_LLMs(self, prompt: str, system_prompt: str, temperature: float = 0.5):
        """
        Inquires the LLM with the given user prompt and system prompt.
        Combines the system and user prompt into a single input string for the llama.cpp server.

        :param prompt: The user's query/prompt.
        :param system_prompt: The system's instructions/context.
        :param temperature: Sampling temperature for controlling randomness.
        :return: The LLM's generated response as a string.
        """
        # Combine system and user prompts into a format suitable for instruction-tuned models
        # and the /v1/completions endpoint. This is a common and often effective approach.
        full_prompt = f"{system_prompt}\n\n{prompt}"

        while True:
            try:
                output = self._generate(
                    prompt=full_prompt,
                    temperature=temperature,
                    # No specific stop tokens added by default, as the original clients
                    # also don't explicitly pass them.
                )
                return output
            except requests.exceptions.RequestException as e:
                print(f"Error occurred during LLM inquiry: {e}")
                print("Retrying after 10 seconds...")
                time.sleep(10)
            except Exception as e:
                print(f"An unexpected error occurred: {e}")
                print("Retrying after 10 seconds...")
                time.sleep(10)