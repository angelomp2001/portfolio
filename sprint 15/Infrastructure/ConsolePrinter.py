# EDA/structure_reporter.py
class ConsolePrinter:
    def __init__(self, datastructureanalyzer):
        self.analyzer = datastructureanalyzer

    def console(self, methods=None):
        # If no methods specified, find all callable methods (except dunder)
        if methods is None:
            methods = [
                name for name in dir(self.analyzer)
                if callable(getattr(self.analyzer, name))
                and not name.startswith("__")
            ]

        for method_name in methods:
            method = getattr(self.analyzer, method_name)
            print(f"\n--- {method_name.upper()} ---")
            result = method()
            print(result)

    