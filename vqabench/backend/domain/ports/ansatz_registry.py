from abc import ABC, abstractmethod
from typing import Optional
from backend.domain.models.ansatz import Ansatz, AnsatzType
from backend.domain.models.hamiltonian import Paradigm


class AnsatzRegistry(ABC):

    @abstractmethod
    def register(self, ansatz: Ansatz) -> None:
        pass

    @abstractmethod
    def get(self, name: str) -> Optional[Ansatz]:
        pass

    @abstractmethod
    def list_all(self) -> list[Ansatz]:
        pass

    @abstractmethod
    def list_by_paradigm(self, paradigm: Paradigm) -> list[Ansatz]:
        pass

    @abstractmethod
    def list_by_type(self, ansatz_type: AnsatzType) -> list[Ansatz]:
        pass

    @abstractmethod
    def exists(self, name: str) -> bool:
        pass

    @abstractmethod
    def unregister(self, name: str) -> None:
        pass