"""
Math engine tool using SymPy - for reliable calculations rather than
letting the LLM guess at arithmetic.
"""

import sympy as sp


def solve_expression(expression: str) -> str:
    try:
        result = sp.simplify(sp.sympify(expression))
        return str(result)
    except Exception as e:
        return f"Could not parse expression: {e}"


def solve_equation(equation: str, variable: str = "x") -> str:
    try:
        var = sp.symbols(variable)
        expr = sp.sympify(equation)
        solutions = sp.solve(expr, var)
        return str(solutions)
    except Exception as e:
        return f"Could not solve equation: {e}"


def first_order_half_life(k: float) -> float:
    return float(sp.log(2) / k)


def calculate_dosage(desired_mg_per_kg: float, patient_weight_kg: float) -> float:
    return desired_mg_per_kg * patient_weight_kg


if __name__ == "__main__":
    print("Test 1 - Simplify expression:")
    print("  2*x + 3*x =", solve_expression("2*x + 3*x"))

    print("\nTest 2 - Solve equation:")
    print("  2*x + 4 - 10 = 0  ->  x =", solve_equation("2*x + 4 - 10", "x"))

    print("\nTest 3 - First-order half-life (k = 0.05 per hour):")
    half_life = first_order_half_life(0.05)
    print(f"  t(1/2) = {half_life:.2f} hours")

    print("\nTest 4 - Weight-based dosage (5 mg/kg for 70 kg patient):")
    dose = calculate_dosage(5, 70)
    print(f"  Dose = {dose} mg")

    print("\nIf you see real calculated values above, this tool is working correctly.")
