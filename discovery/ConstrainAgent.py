# ConstrainAgent/ConstrainAgent.py
import os
from typing import List, Tuple, Dict

import numpy as np
from tqdm import tqdm

from LlamaCPPServerClient import LlamaCPPServerClient
from config import (
    MODEL,
    LLAMA_CPP_SERVER_BASE_URL,
)

from LLMs import (
    ConstrainLLM,
    DomainKnowledgeLLM,
    ConstrainReasoningLLM,
)


def selectClient():
    return LlamaCPPServerClient(LLAMA_CPP_SERVER_BASE_URL, MODEL)


class ConstrainAgent:
    """Base class for constraint agents.

    This class serves as the base class for different types of constraint agents
    that generate constraints between variable pairs.
    """

    def __init__(self): ...
    def run(self) -> np.ndarray: ...


class ConstrainNormalAgent(ConstrainAgent):
    """An agent that leverages domain knowledge from LLMs to generate constraints.

    This agent first obtains domain knowledge through LLM queries, then uses this
    knowledge to construct a constraint matrix that can guide causal discovery
    algorithms.
    """

    def __init__(
        self,
        label: List[str],
        dataset_information: str = None,
        node_information: List[str] = None,
        graph_matrix: np.ndarray = None,
        causal_discovery_algorithm: str = None,
        use_reasoning: bool = False,
    ) -> None:
        """Initializes the ConstrainAgent with dataset and domain information.

        Args:
            label: List of variable names in the causal system.
            dataset_information: Optional description of the dataset context.
            node_information: Optional list of detailed descriptions for each variable.
            graph_matrix: Optional adjacency matrix from preliminary causal discovery.
            causal_discovery_algorithm: Optional name of the algorithm used for initial
                causal discovery.
        """
        self.label = label
        self.dataset_information = dataset_information
        self.node_information = node_information
        self.graph_matrix = graph_matrix
        self.causal_discovery_algorithm = causal_discovery_algorithm

        self.prompt_dict = None  # Stores prompts used for LLM queries
        self.domain_knowledge_dict = None  # Stores domain knowledge responses from LLM

        self.node_num = len(self.label)
        self.use_reasoning = use_reasoning
        # Initialize LLM for generating domain knowledge about causal relationships
        client = selectClient()

        self.domain_knowledge_LLM = DomainKnowledgeLLM(
            client,
            self.label,
            dataset_information=self.dataset_information,
            graph_matrix=self.graph_matrix,
            causal_discovery_algorithm=self.causal_discovery_algorithm,
        )

    def load_domain_knowledge(self, cache_path: str) -> Tuple[Dict, Dict]:
        """Loads cached domain knowledge and prompts from disk.

        Args:
            cache_path: Directory path containing the cached files.

        Returns:
            A tuple containing:
                - prompt dictionary
                - domain knowledge dictionary
        """
        print(
            f"Loading domain knowledge from {cache_path}/prompt_dict{'_with_info' if self.dataset_information else ''}{'_reasoning' if self.use_reasoning else ''}.npy"
        )
        self.prompt_dict = np.load(
            f"{cache_path}/prompt_dict{'_with_info' if self.dataset_information else ''}{'_reasoning' if self.use_reasoning else ''}.npy",
            allow_pickle=True,
        ).item()
        self.domain_knowledge_dict = np.load(
            f"{cache_path}/domain_knowledge_dict{'_with_info' if self.dataset_information else ''}{'_reasoning' if self.use_reasoning else ''}.npy",
            allow_pickle=True,
        ).item()
        return self.prompt_dict, self.domain_knowledge_dict

    def save_domain_knowledge(self, cache_path: str) -> None:
        """Saves domain knowledge and prompts to disk for future use.

        Args:
            cache_path: Directory path to save the cache files.
        """
        if not os.path.exists(cache_path):
            os.makedirs(cache_path)
        np.save(
            f"{cache_path}/prompt_dict{'_with_info' if self.dataset_information else ''}{'_reasoning' if self.use_reasoning else ''}.npy",
            self.prompt_dict,
        )
        np.save(
            f"{cache_path}/domain_knowledge_dict{'_with_info' if self.dataset_information else ''}{'_reasoning' if self.use_reasoning else ''}.npy",
            self.domain_knowledge_dict,
        )

    def generate_domain_knowledge(
        self, use_cache: bool = True, cache_path=None
    ) -> Tuple[Dict, Dict]:
        """Generates or loads domain knowledge for all variable pairs using LLM.

        For each pair of variables, queries the LLM to obtain domain expertise about
        their potential causal relationship.

        Args:
            use_cache: If True, attempts to load cached results before generating new ones.
            cache_path: Directory path for cache files.

        Returns:
            A tuple containing:
                - prompt dictionary
                - domain knowledge dictionary
        """
        if use_cache and cache_path is not None:
            return self.load_domain_knowledge(cache_path)

        self.prompt_dict = {}
        self.domain_knowledge_dict = {}

        # Generate domain knowledge for each directed variable pair
        for i, j in tqdm(
            [
                (i, j)
                for i in range(self.node_num)
                for j in range(self.node_num)
                if i != j
            ],
            desc="Generating domain knowledge",
        ):
            self.prompt_dict[(self.label[i], self.label[j])] = (
                self.domain_knowledge_LLM.generate_prompt(
                    i, j, node_information=self.node_information
                )
            )
            self.domain_knowledge_dict[(self.label[i], self.label[j])] = (
                self.domain_knowledge_LLM.inquiry(temperature=0.5)
            )

        # Cache results if path provided
        if cache_path is not None:
            self.save_domain_knowledge(cache_path)

        return self.prompt_dict, self.domain_knowledge_dict

    def generate_constrain_matrix(self) -> np.ndarray:
        """Generates constraint matrix based on accumulated domain knowledge.

        Uses a separate LLM to analyze the domain knowledge and determine if each
        potential causal relationship is plausible, encoding this as constraints.

        Returns:
            A numpy array representing the constraint matrix where:
                -1 indicates no constraint
                0 indicates the causal relationship is forbidden
                1 indicates the causal relationship is required
        """

        # Initialize LLM for converting domain knowledge into constraint matrix
        #! Must put here, otherwise the domain knowledge dict is empty
        client = selectClient()

        if self.use_reasoning:
            self.constrain_LLM = ConstrainReasoningLLM(
                client,
                self.domain_knowledge_dict,
            )
        else:
            self.constrain_LLM = ConstrainLLM(
                client,
                self.domain_knowledge_dict,
            )
        self.constrain_matrix = np.full((self.node_num, self.node_num), -1)
        for i, j in tqdm(
            [
                (i, j)
                for i in range(self.node_num)
                for j in range(self.node_num)
                if i != j
            ],
            desc="Generating constraint matrix",
        ):
            causal_entity, result_entity = self.label[i], self.label[j]
            self.constrain_LLM.generate_prompt(causal_entity, result_entity)
            self.constrain_LLM.inquiry(temperature=0.8)
            self.constrain_matrix[i, j] = self.constrain_LLM.downstream_processing()
        return self.constrain_matrix

    def run(self, use_cache: bool = True, cache_path=None) -> np.ndarray:
        """Executes the complete constraint generation pipeline.

        First generates/loads domain knowledge, then converts it into a constraint
        matrix.

        Args:
            use_cache: If True, attempts to load cached domain knowledge.
            cache_path: Directory path for cache files.

        Returns:
            Constraint matrix for guiding causal discovery algorithms.
        """
        self.generate_domain_knowledge(use_cache, cache_path)
        self.generate_constrain_matrix()
        return self.constrain_matrix
