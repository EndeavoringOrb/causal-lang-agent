import re
import ollama

class ReactPrompt:
    def __init__(self, model, tooldict, initialprompt = ''):
        """
        :param model: Any LLM interface that supports a .chat(messages:list) → str or .generate(prompt:str) → str
        :param tooldict: dict mapping exactly the action string (e.g. "add[2, 3]") to a Python function
        :param initialprompt: system/user prompt to start the chain
        """
        self.model = model
        self.tooldict = tooldict
        self.initialprompt = initialprompt


    def run(self, prompt, max_steps: int = 10, stop_on_answer: bool = True) -> str:
        """
        Executes the REACT loop until an "Answer:" line appears or max_steps is reached.
        Returns the final answer (or the last model output if no explicit "Answer:" was found).
        """
        history = [ {"role": "system", "content": self.initialprompt}, {"role":"user", "content":prompt} ]
        for step in range(max_steps):
            response = self.model.chat(history)
            history.append({"role": "assistant", "content": response})
            print(history)

            match response[:3]:
                case 'Tho':
                    continue
                case 'Ans':
                    return response
                case 'Act':
                    words = response[8:].split('[') #Action: Add[.....] -> Add, .....]
                    res = self.tooldict[words[0]](words[1][:-1])
                    history.append({"role":"user", "content": "Observation: " + res})
                case _:
                    print("Invalid response: " + response)

        return response


class OllamaModel:
    def __init__(self, model='llama3.2'):
        self.model = model

    def chat(self, messages):
        stream = ollama.chat(
            model=self.model,
            messages=messages,
            stream=True,
        )

        line = ""
        for chunk in stream:
            token = chunk["message"]["content"]
            line += token
            if "\n" in line:
                break  # stop at the first newline
        return line.strip()



# 3) Define your tools
def add(s):
    return str(sum(map(int, s.split(', '))))

def multiply(s):
    args = list(map(int, s.split(', ')))
    return str(args[0] * args[1])

tools = {
    "Add": add,
    "Multiply": multiply,
}

# 4) Initial prompt
initial = """
You are an agent that uses Thought/Action/Observation loops. 
You have access to two actions, Add and Multiply, where Add[a, b] = a + b and Multiply[a, b] = a * b.
Example:
What is (12345+45673) * 23243 ?
Thought: I need to add 12345 and 45673
Action: Add[12345, 45673]
Observation: 58018
Thought: I need to multiply 58018 and 23243
Action: Multiply[58018, 23243]
Observation: 1348512374
Answer: 1348512374

"""

# 5) Instantiate
llm = OllamaModel()
agent = ReactPrompt(model=llm, tooldict=tools, initialprompt=initial)

result = agent.run('what is 12124 * 68389?')
print("Final Answer:", result)
