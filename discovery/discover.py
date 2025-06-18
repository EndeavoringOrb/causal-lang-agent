from discovery.ConstrainAgent import ConstrainNormalAgent
from discovery.CausalDiscovery import (
    causal_discovery,
)
from discovery.visualize import visualize_graph
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

causal_discovery_algorithm = "pc"
# causal_discovery_algorithm = "Exact-Search"
# causal_discovery_algorithm = "DirectLiNGAM"


def load_data_from_csv(filename) -> tuple[np.ndarray, list[str]]:
    """Load the data from the specified CSV file and encode categorical features.

    Args:
        filename: Path to the CSV file.

    Returns:
        tuple:
            - np.ndarray: Scaled numeric values from the dataset (with categorical columns encoded).
            - list[str]: The feature labels for the dataset.
    """
    data = pd.read_csv(filename)
    data = data.dropna()

    # Encode string columns to numbers
    for col in data.select_dtypes(include=["object"]).columns:
        unique_vals = data[col].unique()
        mapping = {val: idx for idx, val in enumerate(unique_vals)}
        data[col] = data[col].map(mapping)

    scaler = StandardScaler()
    values = scaler.fit_transform(data)
    labels = list(data.columns)

    return values, labels


def discover(filename, data_desc, llm_only=False):
    print(f"Loading dataset: {filename}...")
    data, labels = load_data_from_csv(filename)

    print(f"Running {causal_discovery_algorithm} algorithm...")
    adjacency_matrix = causal_discovery(data, labels, method=causal_discovery_algorithm)
    visualize_graph(
        adjacency_matrix,
        labels,
        f"./images/{str(filename).split('/')[-1].strip('.csv')}_{causal_discovery_algorithm}_graph.png",
    )

    print("Running ConstrainAgent...")
    constrain_agent = ConstrainNormalAgent(
        labels,
        graph_matrix=adjacency_matrix,
        causal_discovery_algorithm=causal_discovery_algorithm,
        dataset_information=data_desc,
    )

    constraint_matrix = constrain_agent.run(
        use_cache=False,
        cache_path=f"./cache/Domain_knowledge/{filename.strip('.csv')}/{causal_discovery_algorithm}",
    )
    print("The constraint matrix is:")
    print(constraint_matrix)

    adjacency_matrix_optimized = causal_discovery(
        data,
        labels,
        method=causal_discovery_algorithm,
        constraint_matrix=constraint_matrix,
    )
    print("The optimized adjacency matrix is:")
    print(adjacency_matrix_optimized)

    visualize_graph(
        adjacency_matrix_optimized,
        labels,
        f"./images/{str(filename).split('/')[-1].strip('.csv')}_{causal_discovery_algorithm}_CCAgent.png",
    )
    if llm_only:
        adjacency_matrix_optimized = constraint_matrix
        adjacency_matrix_optimized[adjacency_matrix_optimized == 2] = 0

    return adjacency_matrix_optimized, labels


if __name__ == "__main__":
    discover("matmcd/data/Auto_MPG_data.csv")
