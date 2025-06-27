import pandas as pd
import os

prompts = {
    "identify_common_causes_effect_modifiers": """
You are a data analyst and good at quantitative reasoning. You are required to respond to a quantitative question using the 
provided data. The description and the question can be found below. Please analyze the first 10 rows of the table and write 
python code to analyze the whole table. You should use the DoWhy or causalinference library to build a causal model and perform effect estimation. The steps you should take are: 
1. Identify treatment and effect.
2. Identify which method to use to estimate the effect, for example outcome modeling, propensity scores, difference in differences, instrumental variables, etc.
3. Identify which tools to use. For example, for propensity score matching you could use causalinference CausalModel, and for outcome modelling you could use Dowhy CausalModel with backdoor.linear_regression.
4. Identify variables to adjust for. These could be confounders/common causes, instrumental variables, etc.
5. Identify how (if necessary) to transform the data. For example, in difference in differences, you would need to separate the data based on time and treatment.
6. Write code to estimate the effect and return it.
You do not have to use dowhy or causalinference. Some of the datasets may be structured such that the treatment effect can be calculated from simple pandas methods, and some might be simpler with other libraries. You have access to Dowhy, causalinference, pandas, scipy, econml, and causal-learn. For example, difference in differences does not require any use of dowhy effect estimation.
The returned value of the program should be the answer. After the solution function is written, don't write any more code and enter ```. The solution() function MUST be defined. The general format of the code (in an example of outcome modeling with dowhy) should be
```python
def solution():
    from dowhy import CausalModel
    import pandas as pd

    data = pd.read_csv(filename)

    #filter or transform data as necessary

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
    "original": """You are a data analyst and good at quantitative reasoning. You are required to respond to a quantitative question using the 
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
""",
    "no_discovery": """You are a data analyst and good at quantitative reasoning. You are required to respond to a quantitative question using the 
provided data. The description and the question can be found below. Please analyze the first 10 rows of the table and write 
python code to analyze the whole table. You must use the DoWhy library to build a causal model and perform effect estimation. The returned value of the program should be 
the answer. Write the graph text in gml format. After the solution function is written, don't write any more code and enter ```. The general format of the code should be
```python
def solution():
    from dowhy import CausalModel
    import pandas as pd

    data = pd.read_csv(filename)
    
    # Example graph
    graph = \"\"\"graph [
    directed 1
    node [
        id 1
        label "X"
    ]
    node [
        id 2
        label "Y"
    ]
    node [
        id 3
        label "Z"
    ]
    edge [
        source 1
        target 2
    ]
    edge [
        source 2
        target 3
    ]
]\"\"\"

    model = CausalModel(
        data = data,
        treatment = "treatment_col"
        outcome = "outcome_col"
        graph = graph
    )
    identified_estimand = model.identify_effect()
    answer = model.estimate_effect(identified_estimand, method_name="...")
    return answer.value
```
""",
}


dowhy_est_methods = """
DoWhy: Different estimation methods for causal inference
This is a quick introduction to the DoWhy causal inference library. We will load in a sample dataset and use different methods for estimating the causal effect of a (pre-specified)treatment variable on a (pre-specified) outcome variable.

Method 1: Outcome estimation
Use linear regression.

model = dowhy.CausalModel(
    data=df
    treatment="treatment_col"
    outcome="outcome_col"
    common_causes=[...] #confounders. Be careful to include the right variables here!
)

identified_estimand = model.identify_effect()

causal_estimate_reg = model.estimate_effect(identified_estimand,
        method_name="backdoor.linear_regression",
        test_significance=True)


Method 2: Distance Matching
Define a distance metric and then use the metric to match closest points between treatment and control.

causal_estimate_dmatch = model.estimate_effect(identified_estimand,
                                              method_name="backdoor.distance_matching",
                                              target_units="att",
                                              method_params={'distance_metric':"minkowski", 'p':2}) #euclidean distance

Alternatively, you could use the causalinference library, which supports bias adjustment. 

cm = causalinference.CausalModel(
    Y=med["outcome"].values, 
    D=med["treatment"].values, 
    X=med[[confounders]].values
)

cm.est_via_matching(matches=1, bias_adj=True)

return cm.estimates["matching"]["ate"]

Method 3: Difference in differences
Seperate the data into the treated and control groups at the different time steps.

filter = lambda a, b: df["outcome"].where(df["time"]==a & df["treatment"]==b)

treated_before, treated_after, control_before, control_after = filter(0, 1), filter(1, 1), filter(0, 0), filter(1, 0)

return((treated_after.mean()-treated_before.mean())-(control_after.mean()-control_before.mean()))

Method 4: Propensity Score Matching
We will be using propensity scores to match units in the data.

causal_estimate_match = model.estimate_effect(identified_estimand,
                                              method_name="backdoor.propensity_score_matching",
                                              target_units="atc")

Method 5: Propensity Score Weighting
We will be using (inverse) propensity scores to assign weights to units in the data. DoWhy supports a few different weighting schemes: 1. Vanilla Inverse Propensity Score weighting (IPS) (weighting_scheme=“ips_weight”) 2. Self-normalized IPS weighting (also known as the Hajek estimator) (weighting_scheme=“ips_normalized_weight”) 3. Stabilized IPS weighting (weighting_scheme = “ips_stabilized_weight”)

causal_estimate_ipw = model.estimate_effect(identified_estimand,
                                            method_name="backdoor.propensity_score_weighting",
                                            target_units = "ate",
                                            method_params={"weighting_scheme":"ips_weight"})

Method 6: Instrumental Variable
We will be using the Wald estimator for the provided instrumental variable.

causal_estimate_iv = model.estimate_effect(identified_estimand,
        method_name="iv.instrumental_variable", method_params = {'iv_instrument_name': 'Z0'})

Method 7: Regression Discontinuity
We will be internally converting this to an equivalent instrumental variables problem.

causal_estimate_regdist = model.estimate_effect(identified_estimand,
        method_name="iv.regression_discontinuity",
        method_params={'rd_variable_name':'Z1',
                       'rd_threshold_value':0.5,
                       'rd_bandwidth': 0.15})

"""


example_trace = """ 
Here is an example of correct reasoning and response on one of these problems:

Data Description: The dataset hospital_treatment.csv includes data from a randomized trial conducted by two hospitals. Both of them are conducting randomised trials on a new drug to treat a certain illness. The outcome of interest is days hospitalised. If the treatment is effective, it will lower the amount of days the patient stays in the hospital. For one of the hospitals, the policy regarding the random treatment is to give it to 90% of its patients while 10% get a placebo. The other hospital has a different policy: it gives the drug to a random 10% of its patients and 90% get a placebo. You are also told that the hospital that gives 90% of the true drug and 10% of placebo usually gets more severe cases of the illness to treat. The CSV file contains columns for `hospital` indicating the hospital a patient belongs to, `treatment` signifying if the patient received the new drug or placebo, `severity` reflecting the severity of the illness, and `days` representing the number of days the patient was hospitalized.
First 10 rows of the data:
hospital_treatment:
...

Query: What is the average treatment effect (ATE) of the new drug on the amount of days the patient stays in the hospital? Please choose the variables to adjust for, conduct linear regression, and round your answer to the nearest hundredth. If the new drug reduces the amount of days the patient stays in the hospital, the answer should be negative.

Reasoning: This study concerns 4 variables: treatment, days, hospital, severity. Clearly treatment is the treatment. It says the outcome of interest is days hospitalised, so days is the outcome.
Since the hospitals administer treatments differently, and the hospital doesn't affect the effect other than through treatment, hospital is an instrumental variable. Severity affects the days someone will be hospitalized, and it affects which hospital a patient goes to. Therefore it is a common cause since it influences treatment through hospital.
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
        instruments=["hospital"],
    )

    identified_estimand = model.identify_effect()

    estimate = model.estimate_effect(
        identified_estimand, method_name="backdoor.linear_regression"
    )

    return round(estimate.value, 2)
```

"""


def format_QRData_item(
    benchmark_path,
    item,
    prompt="identify_common_causes_effect_modifiers",
    example=False,
    rows=10,
):
    with open("causal_model_docs.py", "r", encoding="utf-8") as f:
        causal_model_docs = f.read().strip()
    text = f"""{prompts[prompt]}
Here is documentation for the CausalModel class:
```python
{causal_model_docs}
```
Here are some examples of how to implement different causal models.
{dowhy_est_methods}

""".strip()

    if example:
        text += "\n" + example_trace.strip()

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

    answer_start = "```python\ndef solution():\n"

    return text, answer_start
