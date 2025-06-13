import pandas as pd
from causallearn import search
from causallearn.search.ConstraintBased.FCI import fci
from causallearn.search.ConstraintBased.PC import pc
from causallearn.search.ConstraintBased.CDNOD import cdnod
from causallearn.search.ScoreBased.GES import ges
from causallearn.search.ScoreBased.ExactSearch import bic_exact_search

from causallearn.utils.cit import fisherz
from causallearn.utils.GraphUtils import GraphUtils
import matplotlib.pyplot as plt


d = "hospital_treatment"
data = pd.read_csv(f"QRData/QRData/benchmark/data/{d}.csv")

methods = {'fci':fci, 'pc':pc, 'ges':ges, 'bic':bic_exact_search}

labels = list(data.columns)
data = data.to_numpy()


for label, fn in methods.items():
    res = fn(data)
    match label:
        case 'fci':
            pdy = GraphUtils.to_pydot(res[0], labels=labels)
        case 'pc':
            pdy = GraphUtils.to_pydot(res.G, labels=labels)
        case 'ges':
            pdy = GraphUtils.to_pydot(res['G'], labels=labels)


    pdy.write_png(f"graphs/{d}_{label}.png")

