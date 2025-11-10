from typing import Any, Dict, List
from .qiskit_api import QiskitRuntimeAPI
from ..backend_interface import QuantumBackend
import numpy as np


class QiskitBackend(QuantumBackend):
    def create_circuit(self, num_qubits: int) -> Dict[str, Any]:
        pass

    def add_gate(self, gate_type: str, qubits: List[int], **params) -> Dict[str, Any]:
        pass

    def execute_circuit(self) -> Dict[str, Any]:
        pass

    def get_state_vector(self) -> np.ndarray:
        pass

    def get_probabilities(self) -> np.ndarray:
        pass

    def compute_expectation(self, observable: Any) -> float:
        pass

    @property
    def name(self) -> str:
        pass

