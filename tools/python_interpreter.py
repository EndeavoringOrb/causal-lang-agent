import io
import contextlib
import ast
import types
from typing import Dict, Any, List
from tools.base import BaseTool


class PythonInterpreterTool(BaseTool):
    """Enhanced Python code interpreter with controlled imports."""

    # Allowed built-in functions
    SAFE_BUILTINS = {
        "abs",
        "all",
        "any",
        "ascii",
        "bin",
        "bool",
        "bytearray",
        "bytes",
        "callable",
        "chr",
        "classmethod",
        "complex",
        "delattr",
        "dict",
        "dir",
        "divmod",
        "enumerate",
        "filter",
        "float",
        "format",
        "frozenset",
        "getattr",
        "hasattr",
        "hash",
        "hex",
        "id",
        "int",
        "isinstance",
        "issubclass",
        "iter",
        "len",
        "list",
        "map",
        "max",
        "min",
        "next",
        "object",
        "oct",
        "ord",
        "pow",
        "print",
        "property",
        "range",
        "repr",
        "reversed",
        "round",
        "set",
        "setattr",
        "slice",
        "sorted",
        "staticmethod",
        "str",
        "sum",
        "super",
        "tuple",
        "type",
        "vars",
        "zip",
    }

    # Allowed modules and their allowed attributes
    ALLOWED_MODULES = {
        "math": [
            "sqrt",
            "sin",
            "cos",
            "tan",
            "log",
            "log10",
            "exp",
            "pi",
            "e",
            "ceil",
            "floor",
            "fabs",
            "factorial",
            "gcd",
            "degrees",
            "radians",
        ],
        "random": [
            "random",
            "randint",
            "choice",
            "shuffle",
            "sample",
            "uniform",
            "seed",
        ],
        "datetime": ["datetime", "date", "time", "timedelta", "timezone"],
        "json": ["loads", "dumps", "load", "dump"],
        "base64": ["b64encode", "b64decode"],
        "hashlib": ["md5", "sha1", "sha256", "sha512"],
        "urllib.parse": ["quote", "unquote", "urlencode", "parse_qs"],
        "collections": ["Counter", "defaultdict", "namedtuple", "deque"],
        "itertools": [
            "combinations",
            "permutations",
            "product",
            "chain",
            "cycle",
            "repeat",
        ],
        "functools": ["reduce", "partial", "wraps"],
        "re": ["search", "match", "findall", "sub", "split", "compile"],
    }

    # Dangerous keywords/patterns to block
    BLOCKED_PATTERNS = {
        "__import__",
        "__builtins__",
        "__globals__",
        "__locals__",
        "exec",
        "eval",
        "compile",
        "open",
        "file",
        "input",
        "raw_input",
        "reload",
        "execfile",
        "exit",
        "quit",
    }

    def __init__(self):
        self._execution_globals = {}
        self._setup_safe_environment()

    @property
    def name(self) -> str:
        return "python_interpreter"

    @property
    def description(self) -> str:
        return "Execute Python code with controlled imports and safe environment"

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "code": {
                    "type": "python_code",
                    "description": "Python code to execute",
                },
            },
            "required": ["code"],
        }

    def _setup_safe_environment(self):
        """Setup safe execution environment with controlled builtins."""
        safe_builtins = {}

        # Add safe built-in functions
        for name in self.SAFE_BUILTINS:
            if name in __builtins__:
                safe_builtins[name] = __builtins__[name]

        # Add controlled imports
        safe_builtins["__import__"] = self._safe_import

        self._execution_globals = {
            "__builtins__": safe_builtins,
            "__name__": "__main__",
        }

    def _safe_import(self, name: str, globals=None, locals=None, fromlist=(), level=0):
        """Controlled import function that only allows specific modules."""
        if name not in self.ALLOWED_MODULES:
            raise ImportError(f"Import of '{name}' is not allowed")

        # Import the actual module
        module = __import__(name, globals, locals, fromlist, level)

        # Create a restricted module with only allowed attributes
        allowed_attrs = self.ALLOWED_MODULES[name]
        restricted_module = types.ModuleType(name)

        for attr_name in allowed_attrs:
            if hasattr(module, attr_name):
                setattr(restricted_module, attr_name, getattr(module, attr_name))

        return restricted_module

    def _check_code_safety(self, code: str) -> None:
        """Check if code contains dangerous patterns."""
        code_lower = code.lower()

        for pattern in self.BLOCKED_PATTERNS:
            if pattern in code_lower:
                raise ValueError(f"Blocked pattern detected: {pattern}")

        # Parse AST to check for dangerous node types
        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                # Block certain AST node types that could be dangerous
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    # Check if it's an allowed import
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            if alias.name not in self.ALLOWED_MODULES:
                                raise ValueError(
                                    f"Import '{alias.name}' is not allowed"
                                )
                    elif isinstance(node, ast.ImportFrom):
                        if node.module not in self.ALLOWED_MODULES:
                            raise ValueError(
                                f"Import from '{node.module}' is not allowed"
                            )

                        # Check if imported names are allowed
                        allowed_attrs = self.ALLOWED_MODULES.get(node.module, [])
                        for alias in node.names:
                            if alias.name not in allowed_attrs and alias.name != "*":
                                raise ValueError(
                                    f"Import '{alias.name}' from '{node.module}' is not allowed"
                                )

        except SyntaxError as e:
            raise ValueError(f"Syntax error in code: {str(e)}")

    def execute(self, arguments: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Execute Python code in a controlled environment."""
        code = arguments.get("code", "")

        if not code.strip():
            return [{"type": "text", "text": "No code provided"}]

        self._setup_safe_environment()

        try:
            # Check code safety
            self._check_code_safety(code)

            # Capture stdout
            output_buffer = io.StringIO()

            with contextlib.redirect_stdout(output_buffer):
                # Execute the code
                exec(code, self._execution_globals)

            # Get the output
            output = output_buffer.getvalue()

            # Also show any new variables created
            user_vars = {
                k: v
                for k, v in self._execution_globals.items()
                if not k.startswith("__") and k not in ["print"]
            }

            result_parts = []

            if output.strip():
                result_parts.append(f"Output:\n{output.strip()}")

            if user_vars:
                vars_str = "\n".join(f"{k} = {repr(v)}" for k, v in user_vars.items())
                result_parts.append(f"Variables:\n{vars_str}")

            if not result_parts:
                result_parts.append("Code executed successfully (no output)")

            return [{"type": "text", "text": "\n\n".join(result_parts)}]

        except Exception as e:
            return [{"type": "text", "text": f"Execution error: {str(e)}"}]
