import re

formulas = [
    {
        "query_type": "ate",
        "vars": ["X", "V2", "Y"],
        "input": ["P(Y=1 | V2=0)", "P(Y=1 | V2=1)", "P(X=1 | V2=0)", "P(X=1 | V2=1)"],
        "formula": "[P(Y=1|V2=1)-P(Y=1|V2=0)]/[P(X=1|V2=1)-P(X=1|V2=0)]",
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
        "vars": ["X", "Y"],
        "input": ["P(Y=1 | X=0)", "P(Y=1 | X=1)"],
        "formula": "P(Y=1|X=1) - P(Y=1|X=0)",
        "output": "E[Y | do(X = 1)] - E[Y | do(X = 0)]",
    },
]


def normalize(expr):
    return re.sub(r"\s+", "", expr)


def getModelVars(model_output):
    model_vars = []
    for item in model_output["graph"]["nodes"]:
        model_vars.extend(item.keys())
    return model_vars


def isMatch(model_output, formula):
    # Check if the output query matches
    if normalize(model_output["query"]) != normalize(formula["output"]):
        return False

    model_vars = getModelVars(model_output)
    formula_vars = formula["vars"]

    if len(model_vars) != len(formula_vars):
        return False

    # Map formula vars to model vars
    var_map = dict(
        zip(formula_vars, model_vars)
    )  # TODO: make mapping more robust, maybe compare model_output['graph']['given_info'] with formula['input'] and try to match the positions. can probably be moved to helper functino

    # Normalize given info keys
    given_info = {
        normalize(k): v for k, v in model_output["graph"]["given_info"].items()
    }

    # Try to match required inputs
    for expr in formula["input"]:
        # Substitute formula variables with model variables
        substituted = expr
        for k, v in var_map.items():
            substituted = re.sub(rf"\b{k}\b", v, substituted)
        if normalize(substituted) not in given_info:
            return False

    return True


def eval_formula(model_output, formula):
    model_vars = getModelVars(model_output)
    formula_vars = formula["vars"]
    var_map = dict(zip(formula_vars, model_vars))

    # Normalize given info keys
    given_info = {
        normalize(k): v for k, v in model_output["graph"]["given_info"].items()
    }

    # Build local eval environment
    local_env = {}

    for expr in formula["input"]:
        # Replace formula variables with model variable names, i.e. 'P(Y=1|X=0)' -> 'P(Z=1|F=0)' if model_output has Z & F instead of Y & X
        substituted = expr
        for k, v in var_map.items():
            substituted = re.sub(rf"\b{k}\b", v, substituted)
        key = normalize(substituted)
        local_env[normalize(expr)] = given_info[key]

    # Replace the formula's variable names with their mapped counterparts
    eval_formula = formula["formula"]
    for k, v in var_map.items():
        eval_formula = re.sub(rf"\b{k}\b", v, eval_formula)

    # Replace all input expressions in formula with variable names from local_env
    for expr in formula["input"]:
        norm_expr = normalize(expr)
        eval_formula = eval_formula.replace(expr, f"local_env['{norm_expr}']") # TODO: remove this once formulas are cleaned
        eval_formula = eval_formula.replace(norm_expr, f"local_env['{norm_expr}']")

    return eval(eval_formula)


def solve(model_output):
    for formula in formulas:
        if isMatch(model_output, formula):
            return eval_formula(model_output, formula)
    return None


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
