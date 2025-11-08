import logging
import numpy as np
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
        gate_map = {
            "hadamard": "hadamard",
            "h": "hadamard",
            "x": "pauli-x",
            "y": "pauli-y",
            "z": "pauli-z",
            "cnot": "cnot",
            "cx": "cnot",
            "rx": "rx",
            "toffoli": "toffoli",
            "ccx": "toffoli",
            "swap": "swap",
            "ry": "ry",
            "rz": "rz",
            "t": "t-gate",
            "u": "u",
            "unitary": "u",
            "s": "s-gate",
        }
        gate_name = gate_map.get(gate_type.lower(), gate_type)
        endpoint = f"{self.api_base}/gates/{gate_name}"
        request_params = {}

        if len(qubits) == 1:
            request_params["qubit"] = qubits[0]
        elif len(qubits) == 2 and gate_type.lower() in ["cnot", "cx", "swap"]:
            request_params["control"] = qubits[0]
            request_params["target"] = qubits[1]
        elif len(qubits) == 3 and gate_type.lower() in ["toffoli", "ccx"]:
            request_params["control1"] = qubits[0]
            request_params["control2"] = qubits[1]
            request_params["target"] = qubits[2]
        else:
            # todo: future support for multi-qubit gate implementation
            raise ValueError(f"Invalid number of qubits for gate {gate_type}: {qubits}")
        # Unitary gate parameters
        if "theta" in params and "phi" in params and "lambda" in params:
            request_params["theta"] = params["theta"]
            request_params["phi"] = params["phi"]
            request_params["lambda"] = params["lambda"]

        # RX or RY
        if "theta" in params:
            request_params["theta"] = params["theta"]
        # RZ
        if "phi" in params:
            request_params["phi"] = params["phi"]

        response = requests.post(endpoint, params=request_params)
        response.raise_for_status()
        return response.json()

    def execute_circuit(self) -> dict:
        response = requests.post(f"{self.api_base}/simulate/execute")
        response.raise_for_status()
        return response.json()

    def get_state_vector(self) -> np.ndarray:
        response = requests.get(f"{self.api_base}/state/current")
        response.raise_for_status()
        state_data = response.json()
        amplitudes = state_data.get("amplitudes", [])
        state_vector = []
        for amp in amplitudes:
            real = amp.get("real", 0.0)
            imag = amp.get("imag", 0.0)
            state_vector.append(complex(real, imag))
        return np.array(amplitudes, dtype=complex)

    def get_probabilities(self) -> np.ndarray:
        response = requests.get(f"{self.api_base}/state/probabilities")
        response.raise_for_status()
        prob_data = response.json()
        return np.array(prob_data.get("probabilities", []), dtype=float)

    def reset_state(self) -> str:
        response = requests.post(f"{self.api_base}/state/reset")
        response.raise_for_status()
        return response.json()

    def compute_expectation(self, hamiltonian: np.ndarray) -> float:
        """
        For now, we'll compute this client-side:
            1. Get state vector from Java backend
            2. Compute H|ψ> locally
            3. Compute <ψ|H|ψ>
        Later: Add dedicated endpoint to QubitFlow API
        """
        psi = self.get_state_vector()
        H_psi = hamiltonian @ psi
        expectation = np.vdot(psi, H_psi).real
        return float(expectation)

    def clear_circuit(self):
        response = requests.post(f"{self.api_base}/circuit/clear")
        response.raise_for_status()
        return response.json()

    @property
    def name(self) -> str:
        return "JavaBackend"

    def __repr__(self):
        return f"JavaBackend(url={self.base_url})"