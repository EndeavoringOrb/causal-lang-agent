from causallearn.utils.cit import CIT
import pandas as pd
import numpy as np

def conditional_independence(data, xcol, ycol):
    """
    Perform a conditional independence test using Fisher's Z-test.

    Parameters:
    data (DataFrame): The input data containing the variables.
    xcol (str): The name of the first variable.
    ycol (str): The name of the second variable.

    Returns:
    float: The p-value from the Fisher's Z-test.
    """
    cit_obj = CIT(np.array(data), "fisherz") 
    pvals = []
    X = np.array(data[xcol].values)
    Y = np.array(data[ycol].values)
    for slabel in data.columns:
        print(slabel, xcol, ycol)
        if slabel in [xcol, ycol]:
            continue
        S = np.array(data[slabel].values)
        print(X, Y, S)
        pvals.append(cit_obj(X, Y, S))
    return cit_obj(X, Y), min(pvals)



df = pd.read_csv("QRData/benchmark/data/Neuropathic_32.csv")
print(conditional_independence(df, "R lateral elbow pain", "R C5 radiculopathy"))