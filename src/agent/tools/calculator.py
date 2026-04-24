import math

class CalculatorTool:
    name = "calculator"
    description = "Evaluates math expressions. Input must be a valid math expression string e.g. '2 + 2', '10 * 5 / 2', 'sqrt(16)'"

    def run(self, expression: str) -> str:
        try:
            # Create a safe environment containing only the math functions
            safe_dict = {k: v for k, v in math.__dict__.items() if not k.startswith("__")}
            
            # Evaluate the expression using the safe dictionary
            result = eval(expression, {"__builtins__": None}, safe_dict)
            return f"Result: {result}"
        except Exception as e:
            return f"Calculator error: {e}"
