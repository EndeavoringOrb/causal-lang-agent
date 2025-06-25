import numpy as np
import pandas as pd
from Utils.metrics import Metrics
from Utils.CausalDiscovery import causal_discovery
from Utils.data import load_data_from_csv
from Utils.ExactSearchWeighted import bic_exact_search

data, truth, labels = load_data_from_csv("child")
truth = truth.T #first dim outgoing, second dim incoming

A = np.array([
    #  0   1   2   3   4   5   6   7   8   9   10  11  12  13  14  15  16  17  18  19
    [ 0,  0,  1,  0,  0,  0,  0,  0,  0,  0,   0,  1,  0,  0,  0,  1,  0,  0,  0,  0 ],  # BirthAsphyxia
    [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,   0,  0,  0,  0,  0,  0,  0,  0,  0,  0 ],  # HypDistrib
    [ 0,  1,  0,  0,  0,  0,  0,  0,  0,  0,   0,  0,  0,  0,  0,  0,  0,  0,  0,  0 ],  # HypoxiaInO2
    [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  1,   0,  0,  0,  0,  0,  0,  0,  0,  0,  0 ],  # CO2
    [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,   1,  0,  0,  0,  0,  0,  0,  0,  0,  0 ],  # ChestXray
    [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,   0,  0,  1,  0,  0,  0,  0,  0,  0,  0 ],  # Grunting
    [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,   0,  0,  0,  0,  0,  0,  0,  0,  0,  0 ],  # LVHreport
    [ 0,  0,  1,  0,  0,  0,  0,  0,  0,  0,   0,  0,  0,  0,  0,  0,  0,  0,  0,  0 ],  # LowerBodyO2
    [ 0,  0,  1,  0,  0,  0,  0,  0,  0,  0,   0,  0,  0,  0,  0,  0,  0,  0,  0,  0 ],  # RUQO2
    [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,   0,  0,  0,  0,  0,  0,  0,  0,  0,  0 ],  # CO2Report
    [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,   0,  0,  0,  0,  0,  0,  0,  0,  0,  0 ],  # XrayReport
    [ 0,  0,  0,  1,  0,  1,  0,  1,  1,  0,   0,  0,  0,  0,  0,  0,  0,  1,  0,  1 ],  # Disease
    [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,   0,  0,  0,  0,  0,  0,  0,  0,  0,  0 ],  # GruntingReport
    [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,   0,  1,  0,  0,  0,  0,  0,  0,  0,  1 ],  # Age
    [ 0,  0,  0,  0,  0,  0,  1,  0,  0,  0,   0,  0,  0,  0,  0,  0,  0,  0,  0,  0 ],  # LVH
    [ 0,  1,  1,  0,  0,  0,  0,  0,  0,  0,   0,  1,  0,  0,  0,  0,  0,  0,  0,  0 ],  # DuctFlow
    [ 0,  0,  1,  0,  0,  0,  0,  0,  0,  0,   0,  1,  0,  0,  0,  1,  0,  0,  0,  0 ],  # CardiacMixing
    [ 0,  0,  0,  0,  1,  0,  0,  0,  0,  0,   0,  0,  0,  0,  0,  0,  0,  0,  1,  0 ],  # LungParench
    [ 0,  1,  0,  0,  1,  0,  0,  0,  0,  0,   0,  0,  0,  0,  0,  0,  0,  0,  0,  0 ],  # LungFlow
    [ 0,  0,  0,  0,  0,  0,  0,  0,  0,  0,   0,  0,  0,  0,  0,  0,  0,  0,  0,  0 ],  # Sick
], dtype=int)

print("ChatGPT predicted causal relations")
Metrics(A, truth).show_metrics()

print("\nExact search")

inital_cd = causal_discovery(data, labels, 'Exact-Search')
Metrics(inital_cd, truth).show_metrics()

print("\nExact search + Matmcd constraint matrix")
matmcd_style_cd = causal_discovery(data, labels, 'Exact-Search', constraint_matrix=A)



Metrics(matmcd_style_cd, truth).show_metrics()


for w in np.linspace(0, 200, 100):
    ours, _ = bic_exact_search(data, max_parents=1, suggested_graph=A, suggestion_weight=w)

    
    f1 = Metrics(ours, truth).calc_F1score()
    print(f"\nExact search with modified score function, weight = {w}, f1 = {f1:.3f}")


