import pandas as pd
import numpy as np
from dowhy import CausalModel
import json

# Load your data
data = pd.read_csv("QRData/QRData/benchmark/data/hospital_treatment.csv")

graph = """
        digraph {
            push_assigned -> push_delivered;
            push_delivered -> in_app_purchase;
        }
    """
graph = """
graph [
    directed 1
    node [
        id 1
        label "treatment"
    ]
    node [
        id 2
        label "days"
    ]
    node [
        id 3
        label "hospital"
    ]
    node [
        id 4
        label "severity"
    ]
    edge [
        source 1
        target 2
    ]
    edge [
        source 4
        target 2
    ]
    edge [
        source 3
        target 1
    ]
    edge [
        source 4
        target 3
    ]
]
"""



# Create the causal model
model = CausalModel(
    data=data,
    treatment="treatment",  # Actual treatment received
    outcome="days",  # Outcome of interest
    graph=graph,
)



    # Identify the causal estimand
identified_estimand = model.identify_effect()
print(identified_estimand)

# Estimate the effect using Instrumental Variable (IV) regression
estimate = model.estimate_effect(
    identified_estimand, method_name="backdoor.linear_regression"
)

    # Return the estimated LATE, rounded to 2 decimal places
print(round(estimate.value, 2))

# 1) Load the data
df = pd.read_csv("QRData/QRData/benchmark/data/ak91.csv")

# 2) Define the instrument and controls
df["q4"] = (df["quarter_of_birth"] == 4).astype(int)
# One-hot–encode state_of_birth
state_dummies = pd.get_dummies(df["state_of_birth"], prefix="state", drop_first=True)
df = pd.concat([df, state_dummies], axis=1)

# 3) Specify the causal model
model = CausalModel(
    data=df,
    treatment="years_of_schooling",
    outcome="log_wage",
    instruments=["q4"],
    common_causes=["year_of_birth"] + list(state_dummies.columns)
)

# 4) Identify and estimate using two-stage least squares (IV)
identified = model.identify_effect()
iv_estimate = model.estimate_effect(identified, method_name="iv.instrumental_variable")

beta_hat = iv_estimate.value   # this is the effect on log_wage per extra year

# 5) Convert to % wage change:  (exp(β) − 1) × 100
percent_increase = (np.exp(beta_hat) - 1) * 100
print(f"Estimated % increase in wage per extra year of schooling: {percent_increase:.2f}%")
