from discovery.ConstrainAgent import ConstrainNormalAgent
from discovery.CausalDiscovery import (
    causal_discovery,
)
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

causal_discovery_algorithm = "pc"
# causal_discovery_algorithm = "Exact-Search"
# causal_discovery_algorithm = "DirectLiNGAM"


def load_data_from_csv(filename) -> tuple[np.ndarray, list[str]]:
    """Load the data from the specified dataset.

    Args:
        name: The name of the dataset. Must be one of 'Auto_MPG', 'DWD_climate', or 'Sachs'.

    Raises:
        ValueError: If the provided dataset name is not in the list of available options.

    Returns:
        tuple: A tuple containing three elements:
            - np.ndarray: The matrix of values from the dataset.
            - np.ndarray: The ground truth causal relationship matrix for the dataset.
            - list[str]: The feature labels for the dataset.
    """
    data = pd.read_csv(filename)
    scaler = StandardScaler()
    values = scaler.fit_transform(data)
    labels = list(data.columns)
    return values, labels


def discover(filename):
    print(f"Loading dataset: {filename}...")
    data, labels = load_data_from_csv(filename)

    print(f"Running {causal_discovery_algorithm} algorithm...")
    adjacency_matrix = causal_discovery(data, labels, method=causal_discovery_algorithm)

    print("Running ConstrainAgent...")
    constrain_agent = ConstrainNormalAgent(
        labels,
        graph_matrix=adjacency_matrix,
        causal_discovery_algorithm=causal_discovery_algorithm,
        use_reasoning=False,
    )

    constraint_matrix = constrain_agent.run(
        use_cache=False,
        cache_path=f"./cache/Domain_knowledge/{filename.strip('.csv')}/{causal_discovery_algorithm}",
    )

    adjacency_matrix_optimized = causal_discovery(
        data,
        labels,
        method=causal_discovery_algorithm,
        constraint_matrix=constraint_matrix,
    )

    return adjacency_matrix_optimized, labels


if __name__ == "__main__":
    discover("matmcd/data/Auto_MPG_data.csv")
