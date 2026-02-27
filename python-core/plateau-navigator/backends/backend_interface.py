from abc import ABC, abstractmethod
from typing import Dict, Any, List
import numpy as np


class QuantumBackend(ABC):

    @abstractmethod
    def create_circuit(self, num_qubits: int) -> Dict[str, Any]:
        pass

    @abstractmethod
    def add_gate(self, gate_type: str, qubits: List[int], **params) -> Dict[str, Any]:
        pass

    @abstractmethod
    def execute_circuit(self) -> Dict[str, Any]:
        pass

    @abstractmethod
    def get_state_vector(self) -> np.ndarray:
        pass

    @abstractmethod
    def get_probabilities(self) -> np.ndarray:
        pass

    @abstractmethod
    def compute_expectation(self, observable: Any) -> float:
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        pass