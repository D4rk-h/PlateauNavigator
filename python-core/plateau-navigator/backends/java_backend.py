import logging

from backend_interface import QuantumBackend
import requests

class JavaBackend(QuantumBackend):
    """Python client for QubitFlow java backend."""

    def __init__(self, base_url: str = "http://localhost:8080"):
        self.base_url = base_url
        self.api_base = f"{base_url}/api/quantum"
        self._verify_connection()

    def _verify_connection(self):
        try:
            response = requests.get(f"{self.base_url}/api/health")
            response.raise_for_status()
            logging.log("Connected to Java backend successfully.")
        except requests.exceptions.RequestException as e:
            raise ConnectionError(f"Failed to connect to Java backend at port {self.port}") from e

    def create_circuit(self, num_qubits: int) -> dict:
        response = requests.post(
            f"{self.api_base}/circuit/create",
            params={"n_qubit": num_qubits}
        )
        response.raise_for_status()
        return response.json()

    def add_gate(self, gate_type: str, qubits: list, **params) -> dict:
        pass

    def execute_circuit(self) -> dict:
        pass

    def get_state_vector(self) -> 'np.ndarray':
        pass

    def compute_expectation(self, observable: 'Any') -> float:
        pass

    @property
    def name(self) -> str:
        return "JavaBackend"