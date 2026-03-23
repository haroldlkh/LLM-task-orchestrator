from tabular_data_loader import TabularDataLoader
# from image_loader import ImageLoader # Future addition

# Mapping user strings to Python Classes
LOADER_REGISTRY = {
    "tabular": TabularDataLoader,
    # "image": ImageLoader,
}

def get_loader(data_type, temp_dir):
    """Returns the correct loader class based on the data_type string."""
    loader_class = LOADER_REGISTRY.get(data_type.lower())
    if not loader_class:
        available = list(LOADER_REGISTRY.keys())
        raise ValueError(f"Unknown data_type '{data_type}'. Available: {available}")
    
    return loader_class(temp_dir)