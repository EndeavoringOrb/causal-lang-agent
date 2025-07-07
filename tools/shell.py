import io
import os
import importlib
import contextlib
from typing import Any


class PersistentShell:
    def __init__(self, allowed_builtins=None, allowed_imports=None, log=False):
        self._log = log
        self.output = io.StringIO()
        self._allowed_builtins = allowed_builtins or {
            "abs",
            "min",
            "max",
            "len",
            "sum",
            "range",
            "print",
            "int",
            "float",
            "str",
            "bool",
            "list",
            "dict",
            "set",
            "tuple",
            "enumerate",
            "zip",
            "round"
        }
        self._allowed_imports = allowed_imports or {
            "math",
            "random",
            "pandas",
            "sklearn",
            "numpy",
            "statsmodels"
        }
        self._working_dir = os.getcwd()
        self.reset()

    def log(self):
        out = self.output.getvalue()
        if self._log and out:
            print(out, end="")
        self.output.seek(0)
        self.output.truncate(0)
        return out

    def reset(self):
        os.chdir(self._working_dir)  # Reset to stored working directory
        builtins_dict = (
            __builtins__ if isinstance(__builtins__, dict) else __builtins__.__dict__
        )

        def safe_import(name, *args):
            if name.split(".")[0] in self._allowed_imports:
                return importlib.import_module(name)
            raise ImportError(f"Module '{name}' is not allowed.")

        self.namespace: dict[str, Any] = {
            "__builtins__": {
                k: builtins_dict[k]
                for k in self._allowed_builtins
                if k in builtins_dict
            }
        }
        self.namespace["__builtins__"]["__import__"] = safe_import

        for mod in self._allowed_imports:
            try:
                self.namespace[mod] = safe_import(mod)
            except ImportError:
                self.output.write(f"Warning: Module '{mod}' not found.\n")

        return self.log()

    def set_variable(self, name, value):
        self.namespace[name] = value
        return self.log()

    def set_working_directory(self, path: str):
        try:
            os.chdir(path)
            self._working_dir = os.getcwd()
            self.output.write(f"Working directory set to: {self._working_dir}\n")
        except Exception as e:
            self.output.write(f"Failed to set working directory: {e}\n")
        return self.log()

    def get_working_directory(self):
        return self._working_dir

    def run(self, code_str):
        lines = code_str.strip().splitlines()
        temp_namespace = self.namespace.copy()

        try:
            with contextlib.redirect_stdout(self.output):
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        compiled = compile(line, "<input>", "eval")
                        result = eval(compiled, temp_namespace)
                        if result is not None:
                            print(repr(result))
                    except SyntaxError:
                        compiled = compile(line, "<input>", "exec")
                        exec(compiled, temp_namespace)
            self.namespace = temp_namespace
        except Exception as e:
            try:
                compiled = compile(code_str, "<input>", "eval")
                with contextlib.redirect_stdout(self.output):
                    result = eval(compiled, self.namespace)
                    if result is not None:
                        print(repr(result))
            except SyntaxError:
                try:
                    compiled = compile(code_str, "<input>", "exec")
                    with contextlib.redirect_stdout(self.output):
                        exec(compiled, self.namespace)
                except Exception as e:
                    self.output.write(f"Error: {e}\n")
            except Exception as e:
                self.output.write(f"Error: {e}\n")

        return self.log()


if __name__ == "__main__":
    shell = PersistentShell(log=True)

    shell.run("x = 10")
    shell.run("x + 5")
    shell.run("def f():\n  return x * 2")
    shell.run("f()")
    shell.set_variable("y", 99)
    shell.run("y + 1")
    shell.reset()
    shell.run("x")  # Should raise error

    shell.reset()
    shell.run("import math")
    shell.run("math.sqrt(49)")
    shell.run("import os")  # Should raise error

    shell.reset()
    shell.run("import pandas as pd")
    shell.run("df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})")
    shell.run("df")

    shell.reset()
    shell.run("import sklearn.linear_model")

    shell.reset()
    shell.set_working_directory("./QRData/benchmark")
    shell.run("import os")
    shell.run("os.getcwd()")

    shell.reset()
    shell.run("import os")
    shell.run("os.getcwd()")