from abc import ABC, abstractmethod
from typing import Dict, List


class BaseConnector(ABC):
    def __init__(self, config: Dict):
        self.config = config

    @abstractmethod
    def list_objects(self, location: str) -> List[Dict]:
        raise NotImplementedError

    @abstractmethod
    def download_object(self, object_id: str, destination: str) -> None:
        raise NotImplementedError

    @abstractmethod
    def upload_object(self, local_path: str, destination: str, remote_name: str) -> str:
        raise NotImplementedError