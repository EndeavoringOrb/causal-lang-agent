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
from discovery.discover import load_data_from_csv
import os


d = "hospital_treatment"
files = os.listdir("QRData/QRData/benchmark/data")

methods = {'fci':fci, 
           'pc':pc,
          #  'ges':ges
            }

for file in files:        
    print(f"File: {file}")
    data, labels = load_data_from_csv(f"QRData/QRData/benchmark/data/{file}")

    for label, fn in methods.items():
        try:
            res = fn(data)

        except Exception as e:
            print(f"Exception")
            print(f"Method: {label}")
            print(f"Error: {e}")
            continue
        print("Success")
        match label:
            case 'fci':
                pdy = GraphUtils.to_pydot(res[0], labels=labels)
            case 'pc':
                pdy = GraphUtils.to_pydot(res.G, labels=labels)
            case 'ges':
                pdy = GraphUtils.to_pydot(res['G'], labels=labels)


        pdy.write_png(f"graphs/{d}_{label}.png")

