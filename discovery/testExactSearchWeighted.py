import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from discovery.ExactSearch import bic_exact_search
from discovery.ExactSearchWeighted import bic_exact_search as bic_exact_search_weighted
import numpy as np
from matmcd.Utils.metrics import Metrics
from matmcd.Utils.CausalDiscovery import causal_discovery
from matmcd.Utils.data import load_data_from_csv
from utils.timer import Timer

data, truth, labels = load_data_from_csv("child", dirname="matmcd")
truth = truth.T  # first dim outgoing, second dim incoming

A = np.array(
    [
        # 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],  # BirthAsphyxia
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # HypDistrib
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # HypoxiaInO2
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # CO2
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # ChestXray
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],  # Grunting
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # LVHreport
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # LowerBodyO2
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # RUQO2
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # CO2Report
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # XrayReport
        [0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1],  # Disease
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # GruntingReport
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1],  # Age
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # LVH
        [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],  # DuctFlow
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0],  # CardiacMixing
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],  # LungParench
        [0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # LungFlow
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],  # Sick
    ],
    dtype=int,
)

suggestion_weights = [1]

print("ChatGPT predicted causal relations")
Metrics(A, truth).show_metrics()

print("\nExact search + Matmcd constraint matrix")
with Timer():
    matmcd_style_cd = causal_discovery(
        data, labels, "Exact-Search", constraint_matrix=A
    )
Metrics(matmcd_style_cd, truth).show_metrics()

for max_parents in [1, 2]:
    print(f"\nFast? exact search (max_parents={max_parents})")
    with Timer():
        ours, _ = bic_exact_search_weighted(data, max_parents=max_parents)
    Metrics(ours, truth).show_metrics()

    for suggestion_weight in suggestion_weights:
        print(
            f"\nExact search with modified score function, (weight={suggestion_weight}, max_parents={max_parents})"
        )
        with Timer():
            ours, _ = bic_exact_search_weighted(
                data,
                max_parents=max_parents,
                suggested_graph=A,
                suggestion_weight=suggestion_weight,
            )
        Metrics(ours, truth).show_metrics()

    print(f"\nExact search (max_parents={max_parents})")
    with Timer():
        ours, _ = bic_exact_search(data, max_parents=max_parents)
    Metrics(ours, truth).show_metrics()
