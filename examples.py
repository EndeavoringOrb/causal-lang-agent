import pandas as pd
import numpy as np
from dowhy import CausalModel
import causalinference
import json
import causalml

# Load your data
data = pd.read_csv("QRData/benchmark/data/hospital_treatment.csv")

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
import pandas as pd
import numpy as np
from causalml.inference.meta import BaseTRegressor
from xgboost import XGBRegressor

def estimate_att_tlearner(df, treatment_col, outcome_col, confounder_cols):
    """
    Estimate ATT using the CausalML T-learner with XGBRegressor.

    Args:
        df (pd.DataFrame): Input dataframe.
        treatment_col (str): Name of treatment column (binary 0/1).
        outcome_col (str): Name of outcome column.
        confounder_cols (list of str): Names of confounder columns.

    Returns:
        float: Estimated ATT.
    """

    # Extract data
    X = df[confounder_cols].values
    treatment = df[treatment_col].values
    y = df[outcome_col].values

    # Initialize T-learner with separate models for treated and control
    t_learner = BaseTRegressor(learner=XGBRegressor(), control_name=0)

    # Fit the models
    t_learner.fit(X=X, treatment=treatment, y=y)

    # Predict treatment effects
    te, _, _ = t_learner.predict(X=X)

    # Estimate ATT (mean treatment effect among treated units)
    att = np.mean(te[treatment == 1])

    return att

df = pd.read_csv("QRData/benchmark/data/ihdp_0.csv")

print(estimate_att_tlearner(df, "treatment", "y", ["x"+str(i) for i in range(1,26)]))


model = CausalModel(
    data=df,
    treatment="treatment",  
    outcome="y",  
    common_causes=["x"+str(i) for i in range(1,26)],
)
identified_estimand = model.identify_effect()

estimate = model.estimate_effect(
    identified_estimand, method_name="backdoor.propensity_score_weighting", target_units="att"
)

print(round(estimate.value, 2))
exit()



data = pd.read_csv("QRData/benchmark/data/medicine_impact_recovery.csv")
X = ["severity", "age", "sex"]

data['medication'] = data['medication'].astype(bool) # int doesn't work apperently
data = data.assign(**{f: (data[f] - data[f].mean())/data[f].std() for f in X})
cm = causalinference.CausalModel(
    Y=data["recovery"].values, 
    D=data["medication"].values, 
    X=data[["severity", "age", "sex"]].values
)

cm.est_via_matching(matches=1, bias_adj=True)

print(cm.estimates["matching"]["ate"])
    # Return the estimated LATE, rounded to 2 decimal places



# 1) Load the data
df = pd.read_csv("QRData/benchmark/data/ak91.csv")

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
