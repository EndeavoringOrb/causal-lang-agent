from typing import List, Tuple

import numpy as np


class LLMs:
    """Base LLM class that defines the basic interface for LLM interaction.

    This class provides the core functionality for interacting with language models,
    including making inquiries and handling responses.

    Attributes:
        client: The LLM client used for making API calls.
        LLM_answer: The most recent response from the language model.
    """

    def __init__(self, client) -> None:
        """Initializes the LLM class.

        Args:
            client: The LLM client to use for API calls.
        """
        self.client = client  # The LLM client used for making API calls
        self.LLM_answer = None  # The most recent response from the language model

    def inquiry(self) -> str:
        """Makes an inquiry to the language model.

        Args:
            temperature: Sampling temperature for controlling randomness in generation.
                Defaults to 0.5.

        Returns:
            The language model's response as a string.

        Raises:
            ValueError: If prompt or system_prompt is not set.
        """
        if self.prompt is None or self.system_prompt is None:
            raise ValueError(
                "prompt or system_prompt is None. Please call generate_prompt() method first."
            )
        self.LLM_answer = self.client.inquire_LLMs(self.prompt, self.system_prompt)
        return self.LLM_answer

    def inquiry_batched(self, prompts: List[Tuple[str, str]]) -> List[str]:
        """Makes a batched inquiry to the language model.

        Returns:
            The language model's responses as a list of strings.
        """
        self.LLM_answer = self.client.inquire_LLMs_batched(prompts)
        return self.LLM_answer

    def show_prompt(self) -> Tuple[str, str]:
        """Displays the current prompt.

        Returns:
            A tuple containing (user prompt, system prompt).

        Raises:
            ValueError: If prompt is not set.
        """
        if self.prompt is None:
            raise ValueError(
                "prompt is None. Please call generate_prompt() method first."
            )
        return self.prompt, self.system_prompt

    def show_answer(self) -> str:
        """Displays the language model's most recent response.

        Returns:
            The language model's response as a string.

        Raises:
            ValueError: If no inquiry has been made yet.
        """
        if self.LLM_answer is None:
            raise ValueError("LLM_answer is None. Please call inquiry() method first.")
        return self.LLM_answer

    def generate_prompt(self) -> Tuple[str, str]:
        """Abstract method for generating prompts.

        Returns:
            A tuple containing (user prompt, system prompt).

        Raises:
            NotImplementedError: This method must be implemented by subclasses.
        """
        raise NotImplementedError("Subclasses must implement generate_prompt() method")

    def downstream_processing(self):
        """Abstract method for processing language model responses.

        Raises:
            NotImplementedError: This method must be implemented by subclasses.
        """
        raise NotImplementedError(
            "Subclasses must implement downstream_processing() method"
        )


class DomainKnowledgeLLM(LLMs):
    """LLM class for domain knowledge verification.

    This class handles verification of domain knowledge by interacting with language
    models in a specific domain context.

    Attributes:
        labels: List of variable labels.
        dataset_information: Information about the dataset.
        graph_matrix: Causal graph adjacency matrix.
        causal_discovery_algorithm: Name of causal discovery algorithm.
        dataset_prompt: Generated dataset description prompt.
        graph_prompt: Generated graph description prompt.
        prompt: Complete user prompt.
        system_prompt: Complete system prompt.
        LLM_answer: Most recent LLM response.
    """

    def __init__(
        self,
        client,
        labels: List[str],
        dataset_information: str = None,
        graph_matrix: np.ndarray = None,
        causal_discovery_algorithm: str = None,
    ) -> None:
        """Initializes the DomainKnowledgeLLM.

        Args:
            client: The LLM client to use.
            labels: List of variable labels.
            dataset_information: Optional information about the dataset.
            graph_matrix: Optional causal graph adjacency matrix.
            causal_discovery_algorithm: Optional name of causal discovery algorithm.
        """
        super().__init__(client)

        self.labels = labels  # List of variable labels
        self.dataset_information = dataset_information  # Dataset information
        self.graph_matrix = graph_matrix  # Causal graph adjacency matrix
        self.causal_discovery_algorithm = (
            causal_discovery_algorithm  # Name of causal discovery algorithm
        )

        # Prompt components
        self.dataset_prompt = ""
        self.graph_prompt = ""

        # State variables
        self.prompt = None
        self.system_prompt = None
        self.LLM_answer = None

    def generate_graph_prompt(self) -> str:
        """Generates the prompt section describing the causal graph.

        Returns:
            A string containing the graph description prompt.
        """
        num_nodes = self.graph_matrix.shape[0]

        prompt = f"We have conducted the statistical causal discovery with {self.causal_discovery_algorithm} algorithm.\n\n"
        prompt += "The edges and their coefficients of the structural causal model suggested by the statistical causal discovery are as follows:\n"

        # Traverse adjacency matrix to generate edge descriptions
        for i in range(num_nodes):
            for j in range(num_nodes):
                if i == j:
                    continue
                if self.graph_matrix[i, j] == 0:
                    continue
                else:
                    prompt += f"{self.labels[i]} is the cause of {self.labels[j]}.\n"
        prompt += "\n"

        # Add description about target causal relationship
        if self.graph_matrix[self.cause_index, self.result_index] == 1:
            prompt += f"Based on the results above, it seems that changes in {self.cause_entity} directly affect {self.result_entity}.\n\n"
        else:
            prompt += f"Based on the results above, it seems that changes in {self.cause_entity} do not directly affect {self.result_entity}.\n\n"
        return prompt

    def generate_prompt(
        self,
        cause_index: int,
        result_index: int,
        node_information: dict = None,
    ) -> Tuple[str, str]:
        """Generates complete prompt for domain knowledge verification.

        Args:
            cause_index: Index of cause variable.
            result_index: Index of effect variable.
            node_information: Optional additional information about nodes.

        Returns:
            A tuple containing (user prompt, system prompt).
        """
        self.cause_index = cause_index
        self.result_index = result_index
        self.cause_entity = self.labels[self.cause_index]
        self.result_entity = self.labels[self.result_index]

        # Generate dataset description section
        if self.dataset_prompt == "":
            self.dataset_prompt = f"We want to perform causal discovery,"

            if self.dataset_information is not None:
                self.dataset_prompt += f" the summary of dataset: {self.dataset_information}. Considering {', '.join(self.labels)} as variables.\n\n"
            else:
                self.dataset_prompt += (
                    f" considering {', '.join(self.labels)} as variables.\n\n"
                )

        # Generate causal graph description section
        if (
            self.graph_prompt == ""
            and self.graph_matrix is not None
            and self.causal_discovery_algorithm is not None
        ):
            self.graph_prompt = self.generate_graph_prompt()

        if node_information is not None:
            info_prompt = (
                f"In addition, here is the information of {self.cause_entity} and {self.result_entity} from reliable sources.\n"
                f"{node_information[self.cause_entity]}\n\n"
                f"{node_information[self.result_entity]}\n\n"
            )
        else:
            info_prompt = ""

        # Generate task description section
        final_prompt_template = (
            f"Your task is to interpret this result from a domain knowledge perspective "
            f"and determine whether this statistically suggested hypothesis is plausible in "
            f"the context of the domain.\n\n"
            f"Please provide an explanation that leverages your expert knowledge on the causal "
            f"relationship between {self.cause_entity} and {self.result_entity}, "
            f"and assess the correctness of this causal discovery result.\n "
            f"Your response should consider the relevant factors and provide "
            f"a reasonable explanation based on your understanding of the domain."
        )

        # Combine complete prompt
        self.prompt = (
            self.dataset_prompt
            # + self.graph_prompt
            + info_prompt
            + final_prompt_template
        )

        self.system_prompt = f"You are an expert."
        return self.prompt, self.system_prompt


class ConstrainLLM(LLMs):
    """LLM class for background knowledge verification.

    This class handles verification of background knowledge through LLM interactions.

    Attributes:
        domain_knowledge_dict: Dictionary of domain knowledge.
    """

    def __init__(
        self,
        client,
        domain_knowledge_dict: dict,
    ) -> None:
        """Initializes the ConstrainLLM.

        Args:
            client: The LLM client to use.
            domain_knowledge_dict: Dictionary containing domain knowledge.
        """
        super().__init__(client)
        self.domain_knowledge_dict = (
            domain_knowledge_dict  # Dictionary of domain knowledge
        )

    def generate_prompt(self, causal_entity, result_entity) -> Tuple[str, str]:
        """Generates prompt for background knowledge verification.

        Args:
            causal_entity: The potential cause entity.
            result_entity: The potential effect entity.

        Returns:
            A tuple containing (user prompt, system prompt).
        """
        self.prompt = (
            f"Here is the explanation from an expert "
            f"regarding the causal relationship between {causal_entity} and {result_entity}:\n"
            f"{self.domain_knowledge_dict[(causal_entity, result_entity)]}"
            f"Considering the information above, if {causal_entity} is modified, will it have a direct impact on {result_entity} or if {result_entity} is modified, will it have a direct impact on {causal_entity}?\n"
            f"If {causal_entity} causes {result_entity}, answer this question with <0>\n"
            f"If {result_entity} causes {causal_entity}, answer this question with <1>\n"
            f"If there is NO causal relationship, answer this question with <2>\n"
            f"If you are unsure, answer this question with <3>\n"
            f"No answers except these responses are needed.\n"
            f"Your response should be in the following format:\n"
            f"<0> or <1> or <2> or <3>\n"
            f"Please provide your response in the format specified above.\n"
        )
        self.system_prompt = "You are a helpful assistant for causal inference."
        return self.prompt, self.system_prompt

    def downstream_processing(self, answers) -> List[bool]:
        """Processes LLM's response for background knowledge verification.
        """
        final = []
        for answer in answers:
            answer = answer[:-3].strip().strip("<>")
            print("ANSWER FOR CC AGENT:")
            print(answer)
            if answer == "0":
                final.append(0)
            elif answer == "1":
                final.append(1)
            elif answer == "2":
                final.append(2)
            else:
                final.append(3)

        return final
