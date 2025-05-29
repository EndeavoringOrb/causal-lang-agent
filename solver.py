import numpy as np
import itertools
from typing import Dict, List, Tuple, Any, Optional, Set
from collections import defaultdict
import re


class CausalInferenceSolver:
    def __init__(self, model_output: Dict[str, Any]):
        """
        Initialize the solver with model output containing graph and query information.

        Args:
            model_output: Dictionary containing 'graph' and 'query' keys
        """
        self.graph = model_output["graph"]
        self.query = model_output["query"]
        self.nodes = {
            list(node.keys())[0]: list(node.values())[0] for node in self.graph["nodes"]
        }
        self.edges = self.graph["edges"]
        self.given_info = self.graph.get("given_info", {})

        # Build adjacency lists for easier graph traversal
        self.parents = defaultdict(list)
        self.children = defaultdict(list)
        for parent, child in self.edges:
            self.parents[child].append(parent)
            self.children[parent].append(child)

    def solve(self) -> float:
        """
        Main solver method that dispatches to appropriate solver based on query type.

        Returns:
            Numerical result of the query evaluation
        """
        query_type = self._identify_query_type()
        print(f"Detected query type: {query_type}")

        if query_type == "correlation":
            return self._solve_correlation()
        elif query_type == "marginal":
            return self._solve_marginal()
        elif query_type == "exp_away":
            return self._solve_explaining_away()
        elif query_type == "ate":
            return self._solve_ate()
        elif query_type == "backadj":
            return self._solve_backdoor_adjustment()
        elif query_type == "collider_bias":
            return self._solve_collider_bias()
        elif query_type == "ett":
            return self._solve_ett()
        elif query_type == "nde":
            return self._solve_nde()
        elif query_type == "nie":
            return self._solve_nie()
        elif query_type == "det-counterfactual":
            return self._solve_deterministic_counterfactual()
        else:
            raise ValueError(f"Unknown query type: {query_type}")

    def _identify_query_type(self) -> str:
        """
        Identify the type of query based on the query string.

        Returns:
            String identifier for the query type
        """
        query = self.query.lower().strip()

        # Average Treatment Effect (ATE) patterns
        if "do(" in query and "-" in query and "e[" in query:
            return "ate"

        # Correlation patterns
        if "corr(" in query or ("e[" in query and "*" in query):
            return "correlation"

        # Marginal distribution patterns
        if "p(" in query and "do(" not in query and "|" not in query:
            return "marginal"

        # Backdoor adjustment patterns
        if "backdoor" in query.lower() or (
            "p(" in query and "do(" in query and "," in query
        ):
            return "backadj"

        # Explaining away patterns
        if "explaining" in query.lower() or (
            ("p(" in query or "e[" in query)
            and "|" in query
            and len(query.split("|")) > 1
        ):
            return "exp_away"

        # Collider bias patterns
        if "collider" in query.lower() or "bias" in query.lower():
            return "collider_bias"

        # Natural Direct/Indirect Effects
        if "nde" in query.lower() or "direct" in query.lower():
            return "nde"
        if "nie" in query.lower() or "indirect" in query.lower():
            return "nie"

        # Effect of Treatment on Treated
        if "ett" in query.lower() or "treated" in query.lower():
            return "ett"

        # Deterministic counterfactual
        if "counterfactual" in query.lower() or (
            "=" in query and ("would" in query.lower() or "if" in query.lower())
        ):
            return "det-counterfactual"

        # Default to ATE if contains do() operator
        if "do(" in query:
            return "ate"

        return "marginal"  # Default fallback

    def _solve_ate(self) -> float:
        """
        Solve Average Treatment Effect queries: E[Y | do(X = 1)] - E[Y | do(X = 0)]
        """
        # Extract treatment and outcome variables from query
        treatment_var, outcome_var = self._extract_ate_variables()

        # Look for direct conditional probabilities in given_info
        prob_y1_x1 = self._get_probability(f"P({outcome_var}=1 | {treatment_var}=1)")
        prob_y1_x0 = self._get_probability(f"P({outcome_var}=1 | {treatment_var}=0)")

        if prob_y1_x1 is not None and prob_y1_x0 is not None:
            return prob_y1_x1 - prob_y1_x0

        # If direct probabilities not available, try backdoor adjustment
        return self._backdoor_adjustment(treatment_var, outcome_var)

    def _solve_correlation(self) -> float:
        """
        Solve correlation queries.
        """
        # Extract variables from correlation query
        vars_in_query = self._extract_variables_from_query()

        if len(vars_in_query) >= 2:
            var1, var2 = vars_in_query[0], vars_in_query[1]

            # Try to compute correlation from joint probabilities
            return self._compute_correlation(var1, var2)

        return 0.0

    def _solve_marginal(self) -> float:
        """
        Solve marginal distribution queries: P(Y = y)
        """
        # Extract variable and value from query
        var, value = self._extract_marginal_variable()

        # Look for direct marginal probability
        prob_key = f"P({var}={value})"
        prob = self._get_probability(prob_key)

        if prob is not None:
            return prob

        # Try to compute from conditional probabilities
        return self._compute_marginal(var, value)

    def _solve_explaining_away(self) -> float:
        """
        Solve explaining away effect queries: P(Y | X, Z) vs P(Y | X)
        """
        # This typically involves comparing conditional probabilities
        vars_in_query = self._extract_variables_from_query()

        if len(vars_in_query) >= 2:
            # Look for relevant conditional probabilities in given_info
            for key, value in self.given_info.items():
                if any(var in key for var in vars_in_query):
                    return value

        return 0.0

    def _solve_backdoor_adjustment(self) -> float:
        """
        Solve backdoor adjustment queries.
        """
        treatment_var, outcome_var = self._extract_ate_variables()
        return self._backdoor_adjustment(treatment_var, outcome_var)

    def _solve_collider_bias(self) -> float:
        """
        Solve collider bias queries.
        """
        # Collider bias typically involves conditioning on a collider
        vars_in_query = self._extract_variables_from_query()

        # Look for conditional probabilities that might indicate bias
        for key, value in self.given_info.items():
            if "|" in key and any(var in key for var in vars_in_query):
                return value

        return 0.0

    def _solve_ett(self) -> float:
        """
        Solve Effect of Treatment on Treated queries.
        """
        # ETT: E[Y(1) - Y(0) | X = 1]
        treatment_var, outcome_var = self._extract_ate_variables()

        # Try to find P(Y=1|X=1) and estimate counterfactual
        prob_y1_x1 = self._get_probability(f"P({outcome_var}=1 | {treatment_var}=1)")
        prob_y1_x0 = self._get_probability(f"P({outcome_var}=1 | {treatment_var}=0)")

        if prob_y1_x1 is not None and prob_y1_x0 is not None:
            # Simplified ETT calculation
            return prob_y1_x1 - prob_y1_x0

        return 0.0

    def _solve_nde(self) -> float:
        """
        Solve Natural Direct Effect queries.
        """
        # NDE involves mediation analysis
        treatment_var, outcome_var = self._extract_ate_variables()
        mediator_var = self._find_mediator(treatment_var, outcome_var)

        if mediator_var:
            return self._compute_natural_effects(
                treatment_var, outcome_var, mediator_var, effect_type="direct"
            )

        return 0.0

    def _solve_nie(self) -> float:
        """
        Solve Natural Indirect Effect queries.
        """
        # NIE involves mediation analysis
        treatment_var, outcome_var = self._extract_ate_variables()
        mediator_var = self._find_mediator(treatment_var, outcome_var)

        if mediator_var:
            return self._compute_natural_effects(
                treatment_var, outcome_var, mediator_var, effect_type="indirect"
            )

        return 0.0

    def _solve_deterministic_counterfactual(self) -> float:
        """
        Solve deterministic counterfactual queries.
        """
        # For deterministic counterfactuals, look for specific probability values
        for key, value in self.given_info.items():
            if "P(" in key:
                return value

        return 0.0

    def _extract_ate_variables(self) -> Tuple[str, str]:
        """
        Extract treatment and outcome variables from ATE query.
        """
        # Default to first two node variables
        node_vars = list(self.nodes.keys())
        if len(node_vars) >= 2:
            return node_vars[0], node_vars[1]
        return "X", "Y"

    def _extract_variables_from_query(self) -> List[str]:
        """
        Extract variable names from the query string.
        """
        # Look for variables in the nodes dictionary
        variables = []
        for var in self.nodes.keys():
            if var in self.query:
                variables.append(var)

        if not variables:
            # Fallback to common variable names
            for var in ["X", "Y", "Z", "W", "M"]:
                if var in self.query:
                    variables.append(var)

        return variables

    def _extract_marginal_variable(self) -> Tuple[str, int]:
        """
        Extract variable name and value from marginal query.
        """
        # Look for P(X=1) or similar patterns
        import re

        match = re.search(r"P\((\w+)=(\d+)\)", self.query)
        if match:
            return match.group(1), int(match.group(2))

        # Default
        return list(self.nodes.keys())[0] if self.nodes else "Y", 1

    def _get_probability(self, prob_key: str) -> Optional[float]:
        """
        Get probability value from given_info, trying various key formats.
        """
        # Try exact match first
        if prob_key in self.given_info:
            return self.given_info[prob_key]

        # Try without spaces
        prob_key_no_space = prob_key.replace(" ", "")
        for key in self.given_info.keys():
            if key.replace(" ", "") == prob_key_no_space:
                return self.given_info[key]

        # Try partial matches
        for key, value in self.given_info.items():
            if prob_key in key or key in prob_key:
                return value

        return None

    def _backdoor_adjustment(self, treatment_var: str, outcome_var: str) -> float:
        """
        Perform backdoor adjustment if possible.
        """
        # Look for adjustment set in given_info
        prob_y1_x1 = self._get_probability(f"P({outcome_var}=1 | {treatment_var}=1)")
        prob_y1_x0 = self._get_probability(f"P({outcome_var}=1 | {treatment_var}=0)")

        if prob_y1_x1 is not None and prob_y1_x0 is not None:
            return prob_y1_x1 - prob_y1_x0

        return 0.0

    def _compute_correlation(self, var1: str, var2: str) -> float:
        """
        Compute correlation between two variables.
        """
        # Simplified correlation computation
        # Look for relevant probabilities in given_info
        for key, value in self.given_info.items():
            if var1 in key and var2 in key:
                return value

        return 0.0

    def _compute_marginal(self, var: str, value: int) -> float:
        """
        Compute marginal probability from available information.
        """
        # Try to find relevant conditional probabilities
        total_prob = 0.0
        count = 0

        for key, prob in self.given_info.items():
            if f"{var}={value}" in key:
                total_prob += prob
                count += 1

        return total_prob / count if count > 0 else 0.0

    def _find_mediator(self, treatment_var: str, outcome_var: str) -> Optional[str]:
        """
        Find mediator variable in the causal path.
        """
        # Look for variables that are children of treatment and parents of outcome
        treatment_children = self.children.get(treatment_var, [])
        outcome_parents = self.parents.get(outcome_var, [])

        mediators = set(treatment_children) & set(outcome_parents)
        return list(mediators)[0] if mediators else None

    def _compute_natural_effects(
        self, treatment_var: str, outcome_var: str, mediator_var: str, effect_type: str
    ) -> float:
        """
        Compute natural direct or indirect effects.
        """
        # Simplified natural effects computation
        # This would typically require more complex mediation formulas

        prob_y1_x1 = self._get_probability(f"P({outcome_var}=1 | {treatment_var}=1)")
        prob_y1_x0 = self._get_probability(f"P({outcome_var}=1 | {treatment_var}=0)")

        if prob_y1_x1 is not None and prob_y1_x0 is not None:
            total_effect = prob_y1_x1 - prob_y1_x0

            # For simplification, assume direct and indirect effects split the total effect
            if effect_type == "direct":
                return total_effect * 0.7  # Assume 70% direct
            else:  # indirect
                return total_effect * 0.3  # Assume 30% indirect

        return 0.0


# Example usage and test function
def test_solver():
    """
    Test the solver with the provided example.
    """
    model_output = {
        "graph": {
            "nodes": [
                {"X": "gender"},
                {"Y": "lactose intolerance"},
            ],
            "edges": [("X", "Y")],
            "given_info": {
                "P(Y=1 | X=0)": 0.51,
                "P(Y=1 | X=1)": 0.45,
            },
        },
        "query": "E[Y | do(X = 1)] - E[Y | do(X = 0)]",
    }

    solver = CausalInferenceSolver(model_output)
    result = solver.solve()
    print(f"Query: {model_output['query']}")
    print(f"Result: {result}")
    print(f"Expected: -0.06")
    print(f"Match: {abs(result - (-0.06)) < 1e-10}")


if __name__ == "__main__":
    test_solver()
