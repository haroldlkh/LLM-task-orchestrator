import importlib


def load_task_handler_class(task_handler_config: dict):
    if "module" not in task_handler_config or "class" not in task_handler_config:
        raise KeyError(
            "LLM step task_handler config must include 'module' and 'class'"
        )

    module_name = task_handler_config["module"]
    class_name = task_handler_config["class"]

    module = importlib.import_module(module_name)

    if not hasattr(module, class_name):
        raise AttributeError(
            f"LLM task handler module '{module_name}' does not have class '{class_name}'"
        )

    cls = getattr(module, class_name)
    return cls


def build_task_handler(task_handler_config: dict, handler_config: dict | None = None):
    cls = load_task_handler_class(task_handler_config)
    return cls(handler_config or {})