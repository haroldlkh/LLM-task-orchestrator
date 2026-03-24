from .gdrive_connector import GDriveConnector


def get_connector(connector_name: str, config: dict):
    connector_name = connector_name.lower().strip()

    if connector_name == "gdrive":
        return GDriveConnector(config)

    raise ValueError(
        f"Unknown connector '{connector_name}'. Supported connectors: gdrive"
    )