import pandas as pd
import dowhy
from dowhy import CausalModel
from sklearn.ensemble import RandomForestRegressor


def calculate_ate(filepath: str, treatment_col: str, outcome_col: str) -> float:
    """
    Calculate the Average Treatment Effect (ATE) as the difference in mean outcomes
    between treated and untreated groups.

    Parameters:
        treatment_col (str): Column name for treatment indicator (1 for treated, 0 for control).
        outcome_col (str): Column name for the observed outcome.
        filepath (str): Path to the CSV data file.

    Returns:
        float: Estimated Average Treatment Effect (rounded to 2 decimal places).
    """
    df = pd.read_csv(filepath)

    treated = df[df[treatment_col] == 1][outcome_col]
    control = df[df[treatment_col] == 0][outcome_col]

    ate = treated.mean() - control.mean()
    return round(ate, 2)


def calculate_atc(filepath: str, treatment_col: str, outcome_col: str) -> float:
    """
    Calculate the Average Treatment effect on the Control (ATC) using DoWhy.

    Parameters:
        filepath (str): Path to the CSV file.
        treatment_col (str): Column name for the treatment indicator.
        outcome_col (str): Column name for the observed outcome.
        confounders (list[str]):

    Returns:
        float: Estimated ATC (rounded to 2 decimal places).
    """
    # Load the data
    df = pd.read_csv(filepath)
    confounders = [
        col for col in df.columns.tolist() if col not in [treatment_col, outcome_col]
    ]

    # Separate groups
    treated = df[df[treatment_col] == 1]
    control = df[df[treatment_col] == 0]

    # Train model to learn y^(1)
    model = RandomForestRegressor()
    model.fit(treated[confounders], treated[outcome_col])

    # Predict y^(1) for controls
    y1_pred = model.predict(control[confounders])
    y0 = control[outcome_col].values  # actual y for controls is y^(0)

    # Calculate ATC
    atc = (y1_pred - y0).mean()
    return round(atc, 2)


value = calculate_ate("QRData/benchmark/data/ihdp_0.csv", "treatment", "y")
assert value == 4.02, "ate failed"
print(value)
value = calculate_atc("QRData/benchmark/data/ihdp_0.csv", "treatment", "y")
assert value == 4.02, "atc failed"
print(value)
