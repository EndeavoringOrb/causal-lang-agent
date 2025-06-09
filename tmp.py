import pandas as pd
from dowhy import CausalModel

# Load your data
data = pd.read_csv("QRData/benchmark/data/app_engagement_push.csv")

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
        label "push_assigned"
    ]
    node [
        id 2
        label "push_delivered"
    ]
    node [
        id 3
        label "in_app_purchase"
    ]
    edge [
        source 1
        target 2
    ]
    edge [
        source 2
        target 3
    ]
]
"""

# Create the causal model
model = CausalModel(
    data=data,
    treatment="push_delivered",  # Actual treatment received
    outcome="in_app_purchase",  # Outcome of interest
    instruments=["push_assigned"],  # Instrumental variable
    graph=graph,
)

# Identify the causal estimand
identified_estimand = model.identify_effect()

# Estimate the effect using Instrumental Variable (IV) regression
estimate = model.estimate_effect(
    identified_estimand, method_name="iv.instrumental_variable"
)

# Print the estimated LATE, rounded to 2 decimal places
print("Estimated LATE (IV regression):", round(estimate.value, 2))
