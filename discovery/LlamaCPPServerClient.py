import requests
import json
import time
from typing import List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm


class LlamaCPPServerClient:
    def __init__(
        self,
        base_url: str = "http://localhost:55551",
        model: str = "",
        max_parallel_requests: int = 4,
    ) -> None:
        """
        Initialize the LlamaCPPServerClient to interface with a llama.cpp server.

        :param base_url: The base URL of the llama.cpp server API (default is http://localhost:55551).
        :param model: A model identifier (optional, kept for consistency with other clients,
                      llama.cpp server typically doesn't use this directly for /v1/completions).
        """
        self.base_url = base_url.rstrip("/")
        self.model = model  # Stored for consistency, not directly used by the server's completions endpoint.
        self.max_parallel_requests = max_parallel_requests

    def _generate(
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
            # For inquire_LLMs, we need a single string output.
            # Collect streamed content and return it as a complete string.
            full_response = ""
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
                        full_response += content
                    except json.JSONDecodeError as e:
                        print(f"JSON decode error: {e} - Line: {line}")
                        continue
            with open("full_response.txt", "a", encoding="utf-8") as f:
                f.write("\n" * 20 + full_response)
            return full_response
        else:
            return response.json()["choices"][0]["text"]

    def inquire_LLMs(self, prompt: str, system_prompt: str):
        """
        Inquires the LLM with the given user prompt and system prompt.
        Combines the system and user prompt into a single input string for the llama.cpp server.

        :param prompt: The user's query/prompt.
        :param system_prompt: The system's instructions/context.
        :param temperature: Sampling temperature for controlling randomness.
        :return: The LLM's generated response as a string.
        """
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]

        while True:
            try:
                output = self._generate(
                    prompt=messages,
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

    def inquire_LLMs_batched(self, prompts: List[Tuple[str, str]]) -> List[str]:
        """
        Run multiple LLM inquiries in parallel, each with its own (user_prompt, system_prompt).
        """
        results = [None] * len(prompts)

        def run_inquiry(idx: int, user_prompt: str, system_prompt: str):
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
            while True:
                try:
                    result = self._generate(prompt=messages)
                    return (idx, result)
                except requests.exceptions.RequestException as e:
                    print(f"[{idx}] Request error: {e} – retrying in 10s")
                    time.sleep(10)
                except Exception as e:
                    print(f"[{idx}] Unexpected error: {e} – retrying in 10s")
                    time.sleep(10)

        with ThreadPoolExecutor(max_workers=self.max_parallel_requests) as executor:
            futures = [
                executor.submit(run_inquiry, i, user_prompt, system_prompt)
                for i, (user_prompt, system_prompt) in enumerate(prompts)
            ]
            for future in tqdm(
                as_completed(futures), desc="Generating responses", total=len(prompts)
            ):
                idx, output = future.result()
                results[idx] = output

        return results
