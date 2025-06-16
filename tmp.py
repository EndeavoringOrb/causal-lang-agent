import pandas as pd
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


data = json.load(open("QRData/QRData/benchmark/QRData.json"))
for idx, item in enumerate(data):
    item["id"] = idx

# Save to a new file
with open('QRData/QRData/benchmark/QRData_ids.json', 'w') as f:
    json.dump(data, f, indent=2)

exit()

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
    return round(estimate.value, 2)