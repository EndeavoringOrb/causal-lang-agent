import re
from itertools import permutations

formulas = [
    {
        "query_type": "ate",
        "vars": ["X", "V2", "Y"],
        "input": ["P(Y=1 | V2=0)", "P(Y=1 | V2=1)", "P(X=1 | V2=0)", "P(X=1 | V2=1)"],
        "formula": "(P(Y=1|V2=1)-P(Y=1|V2=0))/(P(X=1|V2=1)-P(X=1|V2=0))",
        "output": "E[Y | do(X = 1)] - E[Y | do(X = 0)]",
    },
    {
        "query_type": "ate",
        "vars": ["X", "V3", "Y"],
        "input": [
            "P(V3=1 | X=0)",
            "P(V3=1 | X=1)",
            "P(Y=1 | X=0, V3=0)",
            "P(Y=1 | X=0, V3=1)",
            "P(Y=1 | X=1, V3=0)",
            "P(Y=1 | X=1, V3=1)",
            "P(X=1)",
        ],
        "formula": "(P(V3 = 0|X = 1) - P(V3 = 0|X = 0)) * (P(Y = 1|X = 1,V3 = 0)*P(X = 1)) + (P(V3 = 1|X = 1) - P(V3 = 1|X = 0)) * (P(Y = 1|X = 1,V3 = 1)*P(X = 1))",
        "output": "E[Y | do(X = 1)] - E[Y | do(X = 0)]",
    },
    {
        "query_type": "ate",
        "vars": ["X", "V3", "Y"],
        "input": [
            "P(Y=1 | X=0, V3=0)",
            "P(Y=1 | X=0, V3=1)",
            "P(Y=1 | X=1, V3=0)",
            "P(Y=1 | X=1, V3=1)",
            "P(V3=1)",
        ],
        "formula": "P(V3=0) * (P(Y=1|V3=0,X=1) - P(Y=1|V3=0, X=0)) + P(V3=1) * (P(Y=1|V3=1,X=1) - P(Y=1|V3=1, X=0))",
        "output": "E[Y | do(X = 1)] - E[Y | do(X = 0)]",
    },
    {
        "query_type": "ate",
        "vars": ["X", "Y"],
        "input": ["P(Y=1 | X=0)", "P(Y=1 | X=1)"],
        "formula": "P(Y=1|X=1) - P(Y=1|X=0)",
        "output": "E[Y | do(X = 1)] - E[Y | do(X = 0)]",
    },
    {
        "query_type": "ett",
        "vars": ["X", "Y"],
        "input": ["P(Y=1 | X=0)", "P(Y=1 | X=1)"],
        "formula": "P(Y=1|X=1) - P(Y=1|X=0)",
        "output": "E[Y_{X = 1} - Y_{X = 0} | X = 1]",
    },
    {
        "query_type": "ett",
        "vars": ["X", "V3", "Y"],
        "input": [
            "P(Y=1 | X=0, V3=0)",
            "P(Y=1 | X=0, V3=1)",
            "P(Y=1 | X=1, V3=0)",
            "P(Y=1 | X=1, V3=1)",
            "P(V3=1 | X=0)",
            "P(V3=1 | X=1)",
        ],
        "formula": "P(V3=0|X=1) * (P(Y=1|V3=0,X=1) - P(Y=1|V3=0, X=0)) + P(V3=1|X=1) * (P(Y=1|V3=1,X=1) - P(Y=1|V3=1, X=0))",
        "output": "E[Y_{X = 1} - Y_{X = 0} | X = 1]",
    },
    {
        "query_type": "correlation",
        "vars": ["X", "Y"],
        "input": ["P(X=1)", "P(Y=1, X=0=1)", "P(Y=1, X=1=1)"],
        "formula": "P(X = 1, Y = 1)/P(X = 1) - P(X = 0, Y = 1)/P(X = 0)",
        "output": "P(Y | X)",
    },
    {
        "query_type": "marginal",
        "vars": ["X", "Y"],
        "input": ["P(X=1)", "P(Y=1 | X=0)", "P(Y=1 | X=1)"],
        "formula": "P(Y=1 | X=1)*P(X=1) + P(Y=1 | X=0)*P(X=0)",
        "output": "P(Y)",
    },
    {
        "query_type": "nde",
        "vars": ["X", "V2", "Y"],
        "input": [
            "P(Y=1 | X=0, V2=0)",
            "P(Y=1 | X=0, V2=1)",
            "P(Y=1 | X=1, V2=0)",
            "P(Y=1 | X=1, V2=1)",
            "P(V2=1 | X=0)",
            "P(V2=1 | X=1)",
        ],
        "formula": "P(V2=0|X=0) * (P(Y=1|V2=0,X=1) - P(Y=1|V2=0, X=0)) + P(V2=1|X=0) * (P(Y=1|V2=1,X=1) - P(Y=1|V2=1, X=0))",
        "output": "E[Y_{X=1, V2=0} - Y_{X=0, V2=0}]",
    },
    {
        "query_type": "nie",
        "vars": ["X", "V2", "Y"],
        "input": [
            "P(Y=1 | X=0, V2=0)",
            "P(Y=1 | X=0, V2=1)",
            "P(Y=1 | X=1, V2=0)",
            "P(Y=1 | X=1, V2=1)",
            "P(V2=1 | X=0)",
            "P(V2=1 | X=1)",
        ],
        "formula": "P(Y=1|X=0,V2=0) * (P(V2=0|X=1) - P(V2=0|X=0)) + P(Y=1|X=0,V2=1) * (P(V2=1|X=1) - P(V2=1|X=0))",
        "output": "E[Y_{X=0, V2=1} - Y_{X=0, V2=0}]",
    },
    {
        "query_type": "nie",
        "vars": ["X", "Y"],
        "input": ["P(Y=1 | X=0)", "P(Y=1 | X=1)"],
        "formula": "P(Y=1|X=1) - P(Y=1|X=0)",
        "output": "E[Y_{X=0, V2=1, V3=1} - Y_{X=0, V2=0, V3=0}]",
    },
]


def normalize(expr):
    expr = re.sub(r"\s+", "", expr)  # Remove whitespace

    # Fix malformed equalities
    expr = re.sub(r"\b([A-Za-z0-9_]+)=1=1\b", r"\1=1", expr)
    expr = re.sub(r"\b([A-Za-z0-9_]+)=0=1\b", r"\1=0", expr)

    # Normalize conditional probabilities: sort the conditions
    def normalize_conditionals(match):
        outcome = match.group(1)
        conditions = match.group(2).split(",")
        sorted_conditions = ",".join(sorted(conditions))
        return f"P({outcome}|{sorted_conditions})"

    expr = re.sub(r"P\(([^|]+)\|([^()]+)\)", normalize_conditionals, expr)

    return expr


def getModelVars(model_output):
    model_vars = []
    for item in model_output["graph"]["nodes"]:
        model_vars.extend(item.keys())
    return model_vars


# formula_vars -> given_info_vars
def getVarMap(model_output, formula):
    model_vars = getModelVars(model_output)
    formula_vars = formula["vars"]

    if len(model_vars) != len(formula_vars):
        return None

    given_info = {
        normalize(k): v for k, v in model_output["graph"]["given_info"].items()
    }

    for perm in permutations(model_vars):
        candidate_map = dict(zip(formula_vars, perm))

        all_match = True
        for expr in formula["input"]:
            substituted = expr
            for fvar, mvar in candidate_map.items():
                substituted = re.sub(rf"\b{fvar}\b", mvar, substituted)
            if normalize(substituted) not in given_info:
                all_match = False
                break

        if all_match:
            return candidate_map

    return None


def isMatch(model_output, formula):
    # Check if the output query matches
    if normalize(model_output["query"]) != normalize(formula["output"]):
        return False

    var_map = getVarMap(model_output, formula)
    if not var_map:
        return False

    return True


def eval_formula(model_output, formula):
    var_map = getVarMap(model_output, formula)
    if not var_map:
        raise ValueError("No matching variable map found.")

    given_info = {
        normalize(k): v for k, v in model_output["graph"]["given_info"].items()
    }

    # Normalize formula and substitute var names
    eval_formula = normalize(formula["formula"])
    for k, v in var_map.items():
        eval_formula = re.sub(rf"\b{k}\b", v, eval_formula)

    for expr in given_info:
        norm_expr = normalize(expr)
        eval_formula = eval_formula.replace(norm_expr, f"given_info['{norm_expr}']")

    return eval(eval_formula)


def expand_givens(given_info):
    # Normalize all keys
    given_info = {normalize(k): v for k, v in given_info.items()}

    # Helper to add complements
    def add_complement_probs(info):
        extended = dict(info)
        # Match unconditional probabilities like P(V2=1)
        unconditional_pattern = re.compile(r"P\((\w+)=([01])\)$")
        # Match conditional probabilities like P(V2=1|X=0)
        conditional_pattern = re.compile(r"P\((\w+)=([01])\|(.+)\)$")

        for key, val in info.items():
            m1 = unconditional_pattern.match(key)
            m2 = conditional_pattern.match(key)

            if m1:
                var, value = m1.groups()
                comp_key = f"P({var}={1 - int(value)})"
                norm_comp_key = normalize(comp_key)
                if norm_comp_key not in extended:
                    extended[norm_comp_key] = 1.0 - val

            elif m2:
                var, value, cond = m2.groups()
                comp_key = f"P({var}={1 - int(value)}|{cond})"
                norm_comp_key = normalize(comp_key)
                if norm_comp_key not in extended:
                    extended[norm_comp_key] = 1.0 - val

        return extended

    # Helper to add symmetric joint probabilities
    def add_symmetric_joint_probs(info):
        extended = dict(info)
        pattern = re.compile(r"P\((\w+=\d+),(\w+=\d+)\)")
        for key, val in info.items():
            m = pattern.match(key)
            if m:
                a, b = m.groups()
                sym_key = f"P({b},{a})"
                norm_sym_key = normalize(sym_key)
                if norm_sym_key not in extended:
                    extended[norm_sym_key] = val
        return extended

    # Add complements and symmetric expressions
    given_info = add_complement_probs(given_info)
    given_info = add_symmetric_joint_probs(given_info)

    return given_info


def solve(model_output):
    model_output["graph"]["given_info"] = expand_givens(
        model_output["graph"]["given_info"]
    )
    for formula in formulas:
        if isMatch(model_output, formula):
            return eval_formula(model_output, formula)
    return None


if __name__ == "__main__":
    # Test
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

    result = solve(model_output)
    print("Result:", result)

    model_output = {
        "graph": {
            "nodes": [
                {"X": "zuph"},
                {"Y": "glimx"},
            ],
            "edges": [("X", "Y")],
            "given_info": {"P(X=1)": 0.72, "P(Y=1, X=0)": 0.03, "P(Y=1, X=1)": 0.41},
        },
        "query": "P(Y | X)",
    }

    result = solve(model_output)
    print("Result:", result)
