import json
import requests
from typing import List, Optional, Generator, Union


class LlamaServerClient:
    def __init__(self, model_name: str, base_url: str = "http://localhost:8080"):
        """
        Initialize the LlamaServerClient with a specific model name and base URL.

        :param model_name: The name of the model to use.
        :param base_url: The base URL of the llama-server API.
        """
        self.model_name = model_name
        self.base_url = base_url.rstrip("/")

    def generate(
        self,
        prompt: str,
        stream: bool = False,
        stop: Optional[List[str]] = None,
        **kwargs,
    ) -> Union[str, Generator[str, None, None]]:
        """
        Generate text from the given prompt using the specified model.

        :param prompt: The input prompt string.
        :param stream: Whether to stream the response.
        :param stop: A list of stop sequences.
        :param kwargs: Additional parameters to pass to the API.
        :return: The generated text as a string or a generator yielding strings if streaming.
        """
        url = f"{self.base_url}/v1/completions"
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": stream,
            "stop": stop or [],
            **kwargs,
        }

        response = requests.post(url, json=payload, stream=stream)
        response.raise_for_status()

        if stream:

            def stream_generator():
                for line in response.iter_lines(decode_unicode=True):
                    if line:
                        line = json.loads(line[5:])
                        yield line

            return stream_generator()
        else:
            return response.json()


if __name__ == "__main__":
    # Initialize the client with your model name
    client = LlamaServerClient(model_name="Llama-3.2-1B-Instruct")

    # Generate text without streaming
    output = client.generate(prompt="Once upon a time", stream=False, max_tokens=10)
    print(output)

    # Generate text with streaming
    for chunk in client.generate(prompt="Once upon a time", stream=True):
        print(chunk["choices"][0]["text"], end="")
