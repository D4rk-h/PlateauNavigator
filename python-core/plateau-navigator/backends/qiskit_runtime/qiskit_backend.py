from .utils import serialize_qasm, pauly
import time
from typing import Any, Dict, List, Optional, Tuple, Union
from .qiskit_api import QiskitRuntimeAPI
from ..backend_interface import DVBackend 
import numpy as np


class QiskitBackend(DVBackend):
    """
    DV backend targeting IBM Quantum via the Runtime Estimator primitive.

    Uses the Estimator (not Sampler) for expectation value computation.
    Estimator returns ⟨ψ|H|ψ⟩ directly via Pauli decomposition, avoiding
    shot-noise reconstruction. This is essential for clean gradient variance
    measurements in barren plateau research.

    Session management:
        For VQE runs on real hardware, use as a context manager to batch
        all gradient evaluation jobs into a single IBM Runtime session,
        avoiding per-job queue overhead:

            with QiskitBackend(backend_name="ibm_kyoto", ...) as backend:
                result = vqe.run(initial_params)

    Simulator usage:
        For local testing without IBM credentials, use backend_name="aer_simulator".
        get_state_vector() is only available on simulator backends.
    """

    def __init__(
        self,
        backend_name: str = "ibmq_qasm_simulator",
        api_key: Optional[str] = None,
        crn: Optional[str] = None,
        shots: int = 1024,
        job_timeout: int = 300,
        use_session: bool = False,
    ):
        """
        Args:
            backend_name: IBM backend identifier.
            api_key:      IBM API key (or set IBMQ_API_KEY env var).
            crn:          IBM CRN (or set IBMQ_CRN env var).
            shots:        Number of shots for the Estimator primitive.
            job_timeout:  Seconds to wait for job completion before raising.
            use_session:  If True, creates a Runtime session on first
                          execute_circuit() and reuses it for all subsequent
                          jobs. Required for performant VQE on real hardware.
        """
        self._backend_name = backend_name
        self.shots = shots
        self.job_timeout = job_timeout
        self.use_session = use_session
        self.api = QiskitRuntimeAPI(api_key=api_key, crn=crn)
        self._num_qubits: int = 0
        self._operations: List[Dict[str, Any]] = []
        self._current_qasm: Optional[str] = None
        self._last_result: Optional[Dict[str, Any]] = None
        self._session_id: Optional[str] = None
        # Cache Pauli decomposition — recomputed only when observable changes,
        # not on every VQE iteration
        self._pauli_cache: Optional[List[Tuple[str, complex]]] = None
        self._cached_observable_id: Optional[int] = None
        self._verify_backend()

    def __enter__(self) -> "QiskitBackend":
        if self.use_session:
            session = self.api.create_session(backend=self._backend_name)
            self._session_id = session.get("id")
            print(f"IBM Runtime session opened: {self._session_id}")
        return self

    def __exit__(self, *_) -> None:
        self.close_session()

    def close_session(self) -> None:
        if self._session_id:
            try:
                self.api.close_session(self._session_id)
                print(f"IBM Runtime session closed: {self._session_id}")
            except Exception as e:
                print(f"Warning: could not close session {self._session_id}: {e}")
            finally:
                self._session_id = None

    @property
    def name(self) -> str:
        return f"QiskitBackend({self._backend_name})"

    @property
    def n_qubits(self) -> int:
        return self._num_qubits

    def create_circuit(self, num_qubits: int) -> Dict[str, Any]:
        self._num_qubits = num_qubits
        self._operations = []
        self._current_qasm = None
        self._last_result = None
        return {
            "status": "circuit_created",
            "num_qubits": num_qubits,
            "backend": self._backend_name,
        }

    def add_gate(self, gate_type: str, qubits: List[int], **params) -> Dict[str, Any]:
        if not self._num_qubits:
            raise RuntimeError("Must call create_circuit() before adding gates.")
        gate_key = gate_type.lower()
        if gate_key not in serialize_qasm._GATE_QASM:
            raise ValueError(
                f"Gate '{gate_type}' not supported. "
                f"Supported: {sorted(serialize_qasm._GATE_QASM.keys())}"
            )
        _, expected_qubits, _ = serialize_qasm._GATE_QASM[gate_key]
        if len(qubits) != expected_qubits:
            raise ValueError(
                f"Gate '{gate_type}' requires {expected_qubits} qubit(s), "
                f"got {len(qubits)}."
            )
        if any(q >= self._num_qubits or q < 0 for q in qubits):
            raise IndexError(
                f"Qubit indices {qubits} out of range for "
                f"{self._num_qubits}-qubit circuit."
            )
        self._operations.append({
            "gate_type": gate_key,
            "qubits": qubits,
            "params": params,
        })
        return {"status": "gate_queued", "gate_type": gate_type, "qubits": qubits}

    def execute_circuit(self) -> Dict[str, Any]:
        """
        Build QASM from op queue and submit as an Estimator job.
        The expectation value is computed in compute_expectation() using
        the stored QASM + Pauli-decomposed observable.
        This call only validates connectivity and serializes the circuit.
        Actual job submission happens in compute_expectation().
        """
        if not self._num_qubits:
            raise RuntimeError("No circuit. Call create_circuit() first.")
        self._current_qasm = serialize_qasm._build_qasm(self._num_qubits, self._operations)
        return {
            "status": "circuit_ready",
            "backend": self._backend_name,
            "num_qubits": self._num_qubits,
            "num_gates": len(self._operations),
        }

    def compute_expectation(
        self,
        observable: Union[np.ndarray, List[Tuple[str, complex]]]
    ) -> float:
        """
        Submit an Estimator job and return ⟨ψ|H|ψ⟩.

        If observable is an np.ndarray, it is Pauli-decomposed on first call
        and the result is cached. Subsequent calls with the same observable
        object (by id) skip decomposition — important for VQE where the
        Hamiltonian is fixed but this method is called O(n_params) times
        per gradient step.
        """
        if self._current_qasm is None:
            raise RuntimeError("No circuit ready. Call execute_circuit() first.")
        pauli_terms = self._get_pauli_terms(observable)
        observables_payload = {
            "paulis": [t[0] for t in pauli_terms],
            "coeffs": [t[1].real for t in pauli_terms],
        }
        job_response = self.api.submit_job(
            program_id="estimator",
            backend=self._backend_name,
            params={
                "circuits": [self._current_qasm],
                "observables": [observables_payload],
                "shots": self.shots,
            },
            session_id=self._session_id,
        )
        job_id = job_response.get("id")
        result = self._wait_for_job(job_id)
        self._last_result = result
        evs = result.get("results", [{}])[0].get("data", {}).get("evs", [0.0])
        return float(evs[0] if isinstance(evs, list) else evs)

    def _get_pauli_terms(
        self,
        observable: Union[np.ndarray, List[Tuple[str, complex]]]
    ) -> List[Tuple[str, complex]]:
        """Return Pauli terms, using cache if observable hasn't changed."""
        if isinstance(observable, list):
            return observable
        if not isinstance(observable, np.ndarray):
            raise TypeError(
                f"observable must be np.ndarray or List[Tuple[str, complex]], "
                f"got {type(observable)}."
            )
        obs_id = id(observable)
        if obs_id != self._cached_observable_id or self._pauli_cache is None:
            if not np.allclose(observable, observable.conj().T, atol=1e-10):
                raise ValueError("Observable must be Hermitian (H = H†).")
            self._pauli_cache = pauly._pauli_decompose(observable)
            self._cached_observable_id = obs_id
        return self._pauli_cache

    def get_state_vector(self) -> np.ndarray:
        """
        Available on Aer simulator only via statevector_simulator backend.
        Not available on real hardware — use compute_expectation() directly.
        """
        if "simulator" not in self._backend_name.lower():
            raise NotImplementedError(
                "State vectors are not available on real quantum hardware. "
                "Use compute_expectation() directly, or switch to "
                "'aer_simulator' for local testing."
            )
        raise NotImplementedError(
            "Statevector retrieval requires the Aer local simulator, not the "
            "IBM Runtime API. Instantiate a local AerSimulatorBackend instead."
        )

    def get_probabilities(self) -> np.ndarray:
        """
        Reconstruct bitstring probabilities from last Sampler result.
        Note: QiskitBackend uses the Estimator primitive for VQE.
        This method is provided for circuit debugging only — do not use
        probabilities to reconstruct expectation values in research code.
        """
        if self._last_result is None:
            raise RuntimeError("No results. Call execute_circuit() and compute_expectation() first.")
        counts = (
            self._last_result
            .get("results", [{}])[0]
            .get("data", {})
            .get("counts", {})
        )
        if not counts:
            raise RuntimeError(
                "No counts in last result. This backend uses the Estimator "
                "primitive — counts are not available. Use compute_expectation()."
            )
        total = sum(counts.values())
        probs = np.zeros(2 ** self._num_qubits)
        for bitstring, count in counts.items():
            probs[int(bitstring, 2)] = count / total
        return probs

    def reset_state(self) -> None:
        self._operations = []
        self._current_qasm = None
        self._last_result = None

    def clear_circuit(self) -> None:
        self._operations = []
        self._current_qasm = None

    def get_qasm(self) -> str:
        if self._current_qasm:
            return self._current_qasm
        if self._operations and self._num_qubits:
            return serialize_qasm._build_qasm(self._num_qubits, self._operations)
        raise RuntimeError("No circuit available.")

    def _verify_backend(self) -> None:
        try:
            info = self.api.get_backend(self._backend_name)
            status = info.get("state", {}).get("status", "unknown")
            print(f"Connected to IBM backend: {self._backend_name} (status: {status})")
        except Exception as e:
            raise ConnectionError(
                f"Cannot connect to backend '{self._backend_name}'. "
                f"Verify it exists and your credentials have access. Error: {e}"
            ) from e

    def _wait_for_job(
        self, job_id: str, poll_interval: int = 5
    ) -> Dict[str, Any]:
        deadline = time.time() + self.job_timeout
        while time.time() < deadline:
            info = self.api.get_job(job_id)
            status = info.get("status")
            if status == "Completed":
                return self.api.get_job_results(job_id)
            if status in ("Failed", "Cancelled"):
                raise RuntimeError(
                    f"Job {job_id} {status.lower()}: "
                    f"{info.get('reason', 'no reason provided')}"
                )
            time.sleep(poll_interval)
        try:
            self.api.cancel_job(job_id)
        except Exception:
            pass
        raise TimeoutError(
            f"Job {job_id} did not complete within {self.job_timeout}s."
        )
    
if __name__ == "__main__":
    print("Hola")