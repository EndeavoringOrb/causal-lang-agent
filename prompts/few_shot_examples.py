examples = [
    {
        "question": 357,
        "trace": """ 
Here is an example of correct reasoning and response on one of these problems:

Data Description: The dataset hospital_treatment.csv includes data from a randomized trial conducted by two hospitals. Both of them are conducting randomised trials on a new drug to treat a certain illness. The outcome of interest is days hospitalised. If the treatment is effective, it will lower the amount of days the patient stays in the hospital. For one of the hospitals, the policy regarding the random treatment is to give it to 90% of its patients while 10% get a placebo. The other hospital has a different policy: it gives the drug to a random 10% of its patients and 90% get a placebo. You are also told that the hospital that gives 90% of the true drug and 10% of placebo usually gets more severe cases of the illness to treat. The CSV file contains columns for `hospital` indicating the hospital a patient belongs to, `treatment` signifying if the patient received the new drug or placebo, `severity` reflecting the severity of the illness, and `days` representing the number of days the patient was hospitalized.
First 10 rows of the data:
hospital_treatment:
...

Query: What is the average treatment effect (ATE) of the new drug on the amount of days the patient stays in the hospital? Please choose the variables to adjust for, conduct linear regression, and round your answer to the nearest hundredth. If the new drug reduces the amount of days the patient stays in the hospital, the answer should be negative.

Reasoning: This study concerns 4 variables: treatment, days, hospital, severity. Clearly treatment is the treatment. It says the outcome of interest is days hospitalised, so days is the outcome.
Since the hospitals administer treatments differently, and the hospital doesn't affect the days other than through treatment, hospital is not a confounder. Severity affects the days someone will be hospitalized, and it affects which hospital a patient goes to. Therefore it is a confounder.
In the Dowhy API this would be:
treatment = "treatment",
outcome = "days",
common_causes = ["severity"]
There are no transformations to make to the data. It says to use linear regression, so I should estimate the average treatment effect using Dowhy CausalModel and backdoor.linear_regression.

```python
def solution():
data = pd.load_csv("hospital_treatment.csv")

model = CausalModel(
    data=data,
    treatment="treatment",  
    outcome="days", 
    common_causes=["severity"]
)

identified_estimand = model.identify_effect()

estimate = model.estimate_effect(
    identified_estimand, method_name="backdoor.linear_regression"
)

return round(estimate.value, 2)
```""",
    },
    {
        "question": 362,
        "trace": """
""",
    },
    {
        "question": 312,
        "trace": """
Here is an example of correct reasoning and response on one of these problems:

Data Description: The CSV file ihdp_0.csv contains data obtained from the Infant Health and Development Program (IHDP). The study is designed to evaluate the effect of home visit from specialist doctors on the cognitive test scores of premature infants. The confounders x (x1-x25) correspond to collected measurements of the children and their mothers, including measurements on the child (birth weight, head circumference, weeks born preterm, birth order, first born, neonatal health index, sex, twin status), as well as behaviors engaged in during the pregnancy (smoked cigarettes, drank alcohol, took drugs) and measurements on the mother at the time she gave birth (age, marital status, educational attainment, whether she worked during pregnancy, whether she received prenatal care) and the site (8 total) in which the family resided at the start of the intervention. There are 6 continuous covariates and 19 binary covariates.
First 10 rows of the data:
treatment,y,x1,x2,x3,x4,x5,x6,x7,x8,x9,x10,x11,x12,x13,x14,x15,x16,x17,x18,x19,x20,x21,x22,x23,x24,x25
1,5.59991628549083,-0.528602821749802,-0.343454502314042,1.12855393123738,0.161702527138546,-0.316603181521744,1.29521593563369,1,0,1,0,0,0,0,1,0,1,1,1,1,0,0,0,0,0,0
0,6.87585615601631,-1.73694491077565,-1.80200219804608,0.383827971313067,2.24431955769879,-0.62918919341276,1.29521593563369,0,0,0,1,0,0,1,1,1,1,1,1,1,0,0,0,0,0,0
0,2.99627270536114,-0.807450996140383,-0.20294594178963,-0.360897988611247,-0.879605988141577,0.808706461285914,-0.526555605465598,0,0,0,1,0,0,0,2,0,1,0,1,1,0,0,0,0,0,0
0,1.36620569157546,0.390083024110639,0.596582186338594,-1.85034990845987,-0.879605988141577,-0.0040171696307279,-0.857786794756378,0,0,0,0,0,1,1,2,0,1,0,1,1,0,0,0,0,0,0
0,1.96353813565901,-1.04522850918662,-0.602710005853743,0.0114649913509099,0.161702527138546,0.683672056529507,-0.360940010820207,1,0,0,0,0,1,1,1,0,1,1,1,1,0,0,0,0,0,0
0,4.76209035268695,0.467901119289406,-0.20294594178963,-0.733260968573404,0.161702527138546,0.0585000327474753,1.95767831421525,1,0,0,1,0,0,1,1,0,1,1,1,1,0,0,0,0,0,0
0,6.59404386092927,0.513295008143686,0.596582186338594,0.756190951275224,1.20301104241867,-0.0665343720089312,2.62014069279682,1,0,1,1,0,0,0,1,0,1,1,1,1,0,0,0,0,0,0
0,2.9082345886245,-1.04522850918662,-1.33727610641504,1.12855393123738,-0.879605988141577,0.683672056529507,0.301522367761353,1,0,1,0,0,0,0,2,0,1,1,1,1,0,0,0,0,0,0
0,2.13134648784345,0.748910907434952,0.996346250402707,0.0114649913509099,1.20301104241867,-0.754223598169167,0.798369151697524,1,0,1,1,0,0,0,1,0,1,1,1,1,0,0,0,0,0,0
0,2.60232276243973,-0.396744382696892,-0.602710005853743,0.383827971313067,0.161702527138546,0.621154854151304,-0.526555605465598,0,0,0,1,0,0,0,1,0,1,1,1,1,0,0,0,0,0,0

Query: What is the Average treatment effect of the dataset? Please round to the nearest hundredth.

Reasoning:
```python
def solution():
    import pandas as pd
    import numpy as np
    from causalml.inference.meta import BaseTRegressor
    from xgboost import XGBRegressor

    df = pd.read_csv("QRData/benchmark/data/ihdp_0.csv")

    confounder_cols = [col for col in df.columns if col not in ["treatment", "y"]]

    # Extract data
    X = df[confounder_cols].values
    treatment = df["treatment"].values
    y = df["y"].values

    # Initialize T-learner with separate models for treated and control
    t_learner = BaseTRegressor(learner=XGBRegressor(), control_name=0)

    # Fit the models
    t_learner.fit(X=X, treatment=treatment, y=y)

    # Predict treatment effects
    te = t_learner.predict(X=X)

    # Estimate ATT (mean treatment effect among treated units)
    att = np.mean(te[treatment == 1])

    return round(att, 2)
```""",
    },
]
