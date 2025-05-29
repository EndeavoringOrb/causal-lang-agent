def solve_for_target(known, formulas, target):
    # Keep track of known variables
    known_vars = known.copy()

    # Run until the target is found or no more formulas can be applied
    while target not in known_vars:
        progress = False
        for formula in formulas:
            if formula["output"] in known_vars:
                continue
            # Check if all input variables are known
            if all(var in known_vars for var in formula["input"]):
                # If so, compute the output using eval
                input_vals = {var: known_vars[var] for var in formula["input"]}
                formula_str = formula["formula"]
                result = eval(formula_str, {}, input_vals)
                known_vars[formula["output"]] = result
                progress = True
                print(f"{formula['output']} = {formula['formula']}")
                break  # Re-check the formulas after each new known variable

        # If no progress is made, exit (i.e., we can't solve for the target)
        if not progress:
            return None

    return known_vars[target]


# Example usage
known = {"a": 5, "b": 2}
formulas = [
    {"input": ["a", "b"], "formula": "a * b", "output": "c"},
    {"input": ["a", "c"], "formula": "a * c", "output": "d"},
]

target = "d"
result = solve_for_target(known, formulas, target)

if result is not None:
    print(f"The value of {target} is {result}")
else:
    print(f"Unable to solve for {target}")
