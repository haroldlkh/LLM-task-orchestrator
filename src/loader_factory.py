from tabular_data_loader import TabularDataLoader

LOADER_REGISTRY = {
    "tabular": TabularDataLoader,
}


def get_loader(data_type, temp_dir):
    loader_class = LOADER_REGISTRY.get(data_type.lower())

    if not loader_class:
        available = list(LOADER_REGISTRY.keys())
        raise ValueError(
            f"Unknown data_type '{data_type}'. Available: {available}"
        )

    return loader_class(temp_dir)