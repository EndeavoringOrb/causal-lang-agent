import re


def evaluate_graph(graph, expression):
    info = graph["given_info"]

    def normalize_key(key):
        if "|" not in key:
            return key.strip()
        left, right = key.split("|", 1)
        left = left.strip()
        right = right.strip().rstrip(")")
        conds = [c.strip() for c in right.split(",")]
        conds.sort()
        return f"{left} | {', '.join(conds)})"

    # Normalize all keys upfront
    normalized_info = {}
    for k, v in info.items():
        nk = normalize_key(k)
        normalized_info[nk] = v

    def evaluate_expectation(term):
        term = term.strip()
        match = re.match(r"Y_\{(.*?)\}", term)
        if not match:
            raise ValueError(f"Invalid expectation format: {term}")
        conditions_str = match.group(1)
        cond_parts = [part.strip() for part in conditions_str.split(",")]
        cond_parts.sort()
        cond_str = ", ".join(cond_parts)
        key_str = f"P(Y=1 | {cond_str})"
        if key_str not in normalized_info:
            raise KeyError(f"Missing key: {key_str}")
        return normalized_info[key_str]

    def evaluate_indirect_effect(expr):
        match = re.match(r"INDIRECT_EFFECT\((.*?)\s*->\s*(.*?)\s*->\s*(.*?)\)", expr)
        if not match:
            raise ValueError(f"Invalid INDIRECT_EFFECT format: {expr}")
        x, v, y = match.groups()
        x = x.strip()
        v = v.strip()
        y = y.strip()

        # Get required probabilities
        keys = {
            f"P({y}=1 | {x}=0, {v}=0)",
            f"P({y}=1 | {x}=0, {v}=1)",
            f"P({v}=1 | {x}=0)",
            f"P({v}=1 | {x}=1)",
        }
        probs = {}
        for key in keys:
            nk = normalize_key(key)
            if nk not in normalized_info:
                raise KeyError(f"Missing key: {nk}")
            probs[nk] = normalized_info[nk]

        # compute complement probs
        p_y_x0_v0 = probs[normalize_key(f"P({y}=1 | {x}=0, {v}=0)")]
        p_y_x0_v1 = probs[normalize_key(f"P({y}=1 | {x}=0, {v}=1)")]
        p_v1_x0 = probs[normalize_key(f"P({v}=1 | {x}=0)")]
        p_v1_x1 = probs[normalize_key(f"P({v}=1 | {x}=1)")]
        p_v0_x0 = 1 - p_v1_x0
        p_v0_x1 = 1 - p_v1_x1

        # effect = sum_v P(Y=1 | X=0, V2=v) * (P(V2=v | X=1) - P(V2=v | X=0))
        effect = p_y_x0_v0 * (p_v0_x1 - p_v0_x0) + p_y_x0_v1 * (p_v1_x1 - p_v1_x0)
        return effect

    def parse_expr(expr):
        def replace_indirect_effect(match):
            inside = match.group(0)
            value = evaluate_indirect_effect(inside)
            return str(value)

        expr = re.sub(r"INDIRECT_EFFECT\((.*?)\)", replace_indirect_effect, expr)

        def replace_expectation_diff(match):
            inside = match.group(1)
            parts = inside.split(" - ")
            if len(parts) == 2:
                left = evaluate_expectation(parts[0])
                right = evaluate_expectation(parts[1])
                return f"({left} - {right})"
            else:
                return str(evaluate_expectation(inside))

        expr = re.sub(r"E\[(.*?)\]", replace_expectation_diff, expr)

        def replace_prob(match):
            key = match.group(1).strip()
            # Normalize key inside P(...)
            key_normalized = normalize_key(f"P({key})")
            if key_normalized not in normalized_info:
                raise KeyError(f"Missing key: {key_normalized}")
            return str(normalized_info[key_normalized])

        expr = re.sub(r"P\((.*?)\)", replace_prob, expr)
        return expr

    parsed = None
    try:
        parsed = parse_expr(expression)
        return eval(parsed)
    except Exception as e:
        raise ValueError(
            f"Failed to evaluate: {expression}\nParsed: {parsed}\nError: {e}"
        )


def test():
    graphs = [
        {
            "nodes": [
                {"X": "husband"},
                {"V2": "wife"},
                {"Y": "alarm"},
            ],
            "edges": [("X", "V2"), ("X", "Y"), ("V2", "Y")],
            "given_info": {
                "P(Y=1 | X=0, V2=0)": 0.08,
                "P(Y=1 | X=0, V2=1)": 0.54,
                "P(Y=1 | X=1, V2=0)": 0.41,
                "P(Y=1 | X=1, V2=1)": 0.86,
                "P(V2=1 | X=0)": 0.74,
                "P(V2=1 | X=1)": 0.24,
            },
        },
        {
            "nodes": [
                {"X": "husband"},
                {"Y": "alarm"},
            ],
            "edges": [("X", "Y")],
            "given_info": {
                "P(Y=1 | X=0)": 0.26,
                "P(Y=1 | X=1)": 0.76,
            },
        },
        {
            "nodes": [
                {"X": "husband"},
                {"Y": "alarm"},
            ],
            "edges": [("X", "Y")],
            "given_info": {
                "P(Y=1 | X=0)": 0.26,
                "P(Y=1 | X=1)": 0.76,
                "P(X=1)": 0.77,
            },
        },
        {
            "nodes": [
                {"X": "husband"},
                {"Y": "alarm"},
            ],
            "edges": [("X", "Y")],
            "given_info": {
                "P(Y=1 | X=0)": 0.20,
                "P(Y=1 | X=1)": 0.68,
            },
        },
        {
            "nodes": [
                {"X": "husband"},
                {"V2": "wife"},
                {"Y": "alarm"},
            ],
            "edges": [("X", "V2"), ("X", "Y"), ("V2", "Y")],
            "given_info": {
                "P(Y=1 | X=0, V2=0)": 0.11,
                "P(Y=1 | X=0, V2=1)": 0.60,
                "P(Y=1 | X=1, V2=0)": 0.46,
                "P(Y=1 | X=1, V2=1)": 0.92,
                "P(V2=1 | X=0)": 0.61,
                "P(V2=1 | X=1)": 0.01,
            },
        },
    ]

    assert evaluate_graph(graphs[0], "E[Y_{X=1, V2=0} - Y_{X=0, V2=0}] > 0") == True
    assert evaluate_graph(graphs[1], "E[Y_{X=1} - Y_{X=0}] < 0") == False
    assert (
        evaluate_graph(
            graphs[2], "(P(Y=1 | X=1)*P(X=1) + P(Y=1 | X=0)*(1-P(X=1))) > 0.5"
        )
        == True
    )
    assert evaluate_graph(graphs[3], "(P(Y=1|X=1) - P(Y=1|X=0)) < 0") == False
    assert evaluate_graph(graphs[4], "INDIRECT_EFFECT(X -> V2 -> Y) > 0") == False


if __name__ == "__main__":
    test()
