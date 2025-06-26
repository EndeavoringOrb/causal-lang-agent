import pandas as pd
from dowhy import CausalModel

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
]
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


def solution():
    # Load your data
    data = pd.read_csv("QRData/benchmark/data/app_engagement_push.csv")

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
    print(identified_estimand)

    # Estimate the effect using Instrumental Variable (IV) regression
    estimate = model.estimate_effect(
        identified_estimand, method_name="iv.instrumental_variable"
    )

    # Return the estimated LATE, rounded to 2 decimal places
    return round(estimate.value, 2)
