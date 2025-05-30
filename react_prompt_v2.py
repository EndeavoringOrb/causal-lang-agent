import ollama
import os


def clear_screen():
    # For Windows
    if os.name == "nt":
        _ = os.system("cls")
    # For macOS and Linux
    else:
        _ = os.system("clear")


def printHistory(history):
    clear_screen()
    for message in history:
        print(f"{message['role'][0].upper()}{message['role'][1:]}:")
        print(message["content"].strip())


class OllamaModel:
    def __init__(self, model="gemma3:1b"):
        self.model = model

    def chat(self, messages: list, stop=["\n"]):
        stream = ollama.chat(
            model=self.model,
            messages=messages,
            options={"stop": stop},
            stream=True,
        )

        line = ""
        for chunk in stream:
            token = chunk["message"]["content"]
            line += token
        return line.strip()


class ReactPrompt:
    def __init__(self, model: OllamaModel, tooldict, initialprompt=""):
        """
        :param model: Any LLM interface that supports a .chat(messages:list) → str or .generate(prompt:str) → str
        :param tooldict: dict mapping exactly the action string (e.g. "add[2, 3]") to a Python function
        :param initialprompt: system/user prompt to start the chain
        """
        self.model = model
        self.tooldict = tooldict
        self.initialprompt = initialprompt.strip()

    def run(self, prompt, max_steps: int = 10, stream=False) -> str:
        """
        Executes the REACT loop until an "Answer:" line appears or max_steps is reached.
        Returns the final answer (or the last model output if no explicit "Answer:" was found).
        """
        history = [
            {"role": "system", "content": self.initialprompt},
            {"role": "user", "content": prompt},
        ]
        for step in range(max_steps):
            response = self.model.chat(history)
            if history[-1]["role"] == "assistant":
                history[-1]["content"] = (
                    history[-1]["content"].strip() + "\n" + response.strip() + "\n"
                )
            else:
                history.append(
                    {"role": "assistant", "content": response.strip() + "\n"}
                )
            printHistory(history)

            match response[:3]:
                case "Tho":
                    continue
                case "Act":
                    toolcall = response[8:]
                    words = toolcall.split("[")  # Action: Add[.....] -> Add, .....]
                    if len(words) != 2:
                        res = f"{toolcall} is not a valid tool call."
                    if words[0] == "Answer":
                        return words[1][:-1]
                    elif words[0] not in self.tooldict:
                        res = f"{words[0]} is not a valid tool."
                    else:
                        try:
                            res = self.tooldict[words[0]](words[1][:-1])
                        except Exception:
                            res = f"{words[0]} is not a valid tool call."
                    history.append({"role": "user", "content": "Observation: " + res})
                case _:
                    print("Invalid response: " + response)

        return response


# 3) Define your tools
def add(s):
    return str(sum(map(int, s.split(", "))))


def multiply(s):
    args = list(map(int, s.split(", ")))
    return str(args[0] * args[1])


tools = {
    "Add": add,
    "Multiply": multiply,
}

# 4) Initial prompt
initial = """
You are an agent that uses Thought/Action/Observation loops.
Available actions:
Add[a, b] = a + b
Multiply[a, b] = a * b
Answer[string] used to give your final answer string

Example:
What is (12345+45673) * 23243?
Thought: I need to add 12345 and 45673
Action: Add[12345, 45673]
Observation: 58018
Thought: I need to multiply 58018 and 23243
Action: Multiply[58018, 23243]
Observation: 1348512374
Action: Answer[1348512374]
"""

# 5) Instantiate
llm = OllamaModel("gemma3:1b")
agent = ReactPrompt(model=llm, tooldict=tools, initialprompt=initial)

result = agent.run("what is 12124 * 68389?")
print("Final Answer:", result)
