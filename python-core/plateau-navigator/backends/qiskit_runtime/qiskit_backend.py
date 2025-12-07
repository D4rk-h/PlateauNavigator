import time
from typing import Any, Dict, List, Optional, Callable
from .qiskit_api import QiskitRuntimeAPI
from ..backend_interface import QuantumBackend
from ansatz_translation.ansatz_to_qasm import AnsatzToQasm
import numpy as np


class QiskitBackend(QuantumBackend):
    def __init__(self, backend_name: str = "ibmq_qasm_simulator",
                 api_key: Optional[str] = None,
                 crn: Optional[str] = None,
                 shots: int = 1024):
        self.backend_name = backend_name
        self.shots = shots
        self.api = QiskitRuntimeAPI(api_key=api_key, crn=crn)
        self.num_qubits = 0
        self.qasm_converter: Optional[AnsatzToQasm] = None
        self.current_qasm: Optional[str] = None
        self.last_result: Optional[Dict[str, Any]] = None
        self._verify_backend()

    def _verify_backend(self):
        try:
            backend_info = self.api.get_backend(self.backend_name)
            print(f"Connected to IBM Quantum backend: {self.backend_name}")
            print(f"Status: {backend_info.get('state', {}).get('status', 'unknown')}")
        except Exception as e:
            raise ConnectionError(
                f"Could not connect to backend '{self.backend_name}'. "
                f"Check that it exists and you have access. Error: {e}"
            )

    def create_circuit(self, num_qubits: int) -> Dict[str, Any]:
        self.num_qubits = num_qubits
        self.qasm_converter = AnsatzToQasm(num_qubits)
        return {
            "status": "circuit_created",
            "num_qubits": num_qubits,
            "backend": self.backend_name
        }

    def add_gate(self, gate_type: str, qubits: List[int], **params) -> Dict[str, Any]:
        if self.qasm_converter is None:
            raise RuntimeError("Must call create_circuit() before adding gates")

        self.qasm_converter.add_operation(gate_type, qubits, **params)
        return {
            "status": "gate_added",
            "gate_type": gate_type,
            "qubits": qubits
        }

    def execute_circuit(self) -> Dict[str, Any]:
        if self.qasm_converter is None:
            raise RuntimeError("No circuit to execute. Call create_circuit() first.")
        self.current_qasm = self.qasm_converter.to_qasm(include_measurements=True)
        print(f"Submitting job to {self.backend_name}...")
        print(f"Circuit: {self.num_qubits} qubits, {len(self.qasm_converter.operations)} gates")
        job_response = self.api.submit_job(
            program_id="sampler",
            backend=self.backend_name,
            params={
                "circuits": [self.current_qasm],
                "shots": self.shots
            }
        )
        job_id = job_response.get("id")
        print(f"Job submitted: {job_id}")
        result = self._wait_for_job(job_id)
        self.last_result = result
        return {
            "status": "completed",
            "job_id": job_id,
            "backend": self.backend_name,
            "shots": self.shots,
            "result": result
        }

    def _wait_for_job(self, job_id: str, timeout: int = 300, poll_interval: int = 5) -> Dict[str, Any]:
        start_time = time.time()

        while time.time() - start_time < timeout:
            job_info = self.api.get_job(job_id)
            status = job_info.get("status")
            print(f"Job status: {status}")
            if status == "Completed":
                results = self.api.get_job_results(job_id)
                return results
            elif status in ["Failed", "Cancelled"]:
                raise RuntimeError(f"Job {job_id} {status.lower()}: {job_info.get('reason', 'unknown')}")
            time.sleep(poll_interval)
        raise TimeoutError(f"Job {job_id} did not complete within {timeout} seconds")

    def get_state_vector(self) -> np.ndarray:
        if "simulator" not in self.backend_name.lower():
            raise NotImplementedError(
                "State vectors are not available on real quantum hardware. "
                "Use get_probabilities() instead, or switch to a simulator backend."
            )
        # todo: Need to use a different primitive (Estimator) and modify the job submission.
        raise NotImplementedError(
            "State vector retrieval not yet implemented. "
            "Use the Estimator primitive with statevector simulator."
        )

    def get_probabilities(self) -> np.ndarray:
        if self.last_result is None:
            raise RuntimeError("No execution results available. Call execute_circuit() first.")
        # todo: adjust this to api response
        counts = self.last_result.get("results", [{}])[0].get("data", {}).get("counts", {})
        total_shots = sum(counts.values())
        probabilities = np.zeros(2 ** self.num_qubits)
        for bitstring, count in counts.items():
            index = int(bitstring, 2)
            probabilities[index] = count / total_shots
        return probabilities

    def compute_expectation(self, observable: Any) -> float:
        # todo: enhance method implementing a better Hamiltonian decomp
        if isinstance(observable, np.ndarray):
            probs = self.get_probabilities()
            diagonal = np.diag(observable)
            expectation = np.dot(probs, diagonal)
            return float(expectation)
        else:
            raise NotImplementedError(
                "Observable must be a numpy array. "
                "Pauli string observables not yet supported."
            )

    def reset_state(self):
        if self.qasm_converter:
            self.qasm_converter.reset()

    def clear_circuit(self):
        if self.qasm_converter:
            self.qasm_converter.reset()
        self.current_qasm = None

    @property
    def name(self) -> str:
        return f"QiskitBackend({self.backend_name})"

    def execute_ansatz(self, ansatz_fn: Callable, params: np.ndarray, hamiltonian: np.ndarray) -> float:
        self.clear_circuit()
        self.reset_state()
        ansatz_fn(self, params)
        self.execute_circuit()
        energy = self.compute_expectation(hamiltonian)
        return energy

    def get_qasm(self) -> str:
        if self.current_qasm:
            return self.current_qasm
        elif self.qasm_converter:
            return self.qasm_converter.to_qasm(include_measurements=True)
        else:
            raise RuntimeError("No circuit available")