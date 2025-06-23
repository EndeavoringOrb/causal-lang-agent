import pandas as pd
import os

prompts = {"identify_common_causes_effect_modifiers":"""
You are a data analyst and good at quantitative reasoning. You are required to respond to a quantitative question using the 
provided data. The description and the question can be found below. Please analyze the first 10 rows of the table and write 
python code to analyze the whole table. You must use the DoWhy library to build a causal model and perform effect estimation. The steps you should take are: 
1. Identify treatment, effect, common causes, instruments, and effect modifiers. Common causes are variables that affect the treatment and outcome. Instruments affect the treatment, and effect modifiers affect the outcome.
2. Identify any transformations to the data or additional steps, such as conditioning.
2. Construct a DoWhy CausalModel to perform effect estimation
4. Use DoWhy methods to compute the relevant quantity and return it.
The returned value of the program should be the answer. After the solution function is written, don't write any more code and enter ```. The general format of the code should be
```python
def solution():
    from dowhy import CausalModel
    import pandas as pd

    data = pd.read_csv(filename)

    model = CausalModel(
        data = data,
        treatment = "treatment_col"
        outcome = "outcome_col"
        common_causes = ["causes"]
        instruments = ["instruments"]


    )
    identified_estimand = model.identify_effect()
    answer = model.estimate_effect(identified_estimand, method_name=...)
    return answer.value
```
""",

"original":"""You are a data analyst and good at quantitative reasoning. You are required to respond to a quantitative question using the 
provided data. The description and the question can be found below. Please analyze the first 10 rows of the table and write 
python code to analyze the whole table. You must use the DoWhy library to build a causal model and perform effect estimation. The returned value of the program should be 
the answer. The path to the causal graph is graph.gml. After the solution function is written, don't write any more code and enter ```. The general format of the code should be
```python
def solution():
    from dowhy import CausalModel
    import pandas as pd

    data = pd.read_csv(filename)

    model = CausalModel(
        data = data,
        treatment = "treatment_col"
        outcome = "outcome_col"
        graph = "graph.gml"
    )
    identified_estimand = model.identify_effect()
    answer = model.estimate_effect(identified_estimand, method_name=...)
    return answer.value
```
"""

}

example_trace = """ 
Here is an example of correct reasoning and response on one of these problems:

Data Description: The dataset hospital_treatment.csv includes data from a randomized trial conducted by two hospitals. Both of them are conducting randomised trials on a new drug to treat a certain illness. The outcome of interest is days hospitalised. If the treatment is effective, it will lower the amount of days the patient stays in the hospital. For one of the hospitals, the policy regarding the random treatment is to give it to 90% of its patients while 10% get a placebo. The other hospital has a different policy: it gives the drug to a random 10% of its patients and 90% get a placebo. You are also told that the hospital that gives 90% of the true drug and 10% of placebo usually gets more severe cases of the illness to treat. The CSV file contains columns for `hospital` indicating the hospital a patient belongs to, `treatment` signifying if the patient received the new drug or placebo, `severity` reflecting the severity of the illness, and `days` representing the number of days the patient was hospitalized.
First 10 rows of the data:
hospital_treatment:
...

Query: What is the average treatment effect (ATE) of the new drug on the amount of days the patient stays in the hospital? Please choose the variables to adjust for, conduct linear regression, and round your answer to the nearest hundredth. If the new drug reduces the amount of days the patient stays in the hospital, the answer should be negative.

Reasoning: This study concerns 4 variables: treatment, days, hospital, severity. Clearly treatment is the treatment. It says the outcome of interest is days hospitalised, so days is the outcome.
Since the hospitals administer treatments differently, hospital is an instrumental variable. Severity affects the days someone will be hospitalized, and it affects which hospital a patient goes to. Therefore it is a common cause since it influences treatment through hospital.
There are no transformations to make to the data. I need to estimate the average treatment effect using Dowhy CausalModel and backdoor.linear_regression.

Code: 
```python
def solution():
    data = pd.load_csv("hospital_treatment.csv")

    model = CausalModel(
        data=data,
        treatment="treatment",  
        outcome="days", 
        common_causes=["severity"],
        instruments=["hospital"]
    )

    identified_estimand = model.identify_effect()

    estimate = model.estimate_effect(
        identified_estimand, method_name="backdoor.linear_regression"
    )

    return round(estimate.value, 2)
```

"""


def format_QRData_item(benchmark_path, item, prompt = "identify_common_causes_effect_modifiers", example=False, rows=10):
    with open("causal_model_docs.py", "r", encoding="utf-8") as f:
        causal_model_docs = f.read().strip()
    text = f"""{prompt}
Here is documentation for the CausalModel class:
```python
{causal_model_docs}
```""".strip()
    
    if example:
        text += example_trace

    text += "\nData Description:\n"
    text += item["data_description"]

    text += f"\nFirst {rows} rows of the data:\n"
    for file_name in item["data_files"]:
        text += file_name.strip(".csv") + ":\n"
        df = pd.read_csv(os.path.join(benchmark_path, f"data/{file_name}"))
        df = df.sample(frac=1, random_state=42)  # Shuffle the rows
        text += str(df.head(rows)).strip() + "\n"

    text += "\nQuestion:\n"
    text += item["question"].strip()

    text += "\nResponse\n```python\ndef solution():\n"

    return text