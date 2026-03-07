
from typing import Any, Dict, List, Optional, Tuple, Union
from ..backend_interface import DVBackend
import numpy as np
from qiskit import QuantumCircuit
from qiskit.primitives import StatevectorEstimator
from qiskit.quantum_info import SparsePauliOp, Statevector
from utils import pauly

_GATE_DISPATCH: Dict[str, Tuple[str, int, List[str]]] = {
    "h":       ("h",    1, []),
    "x":       ("x",    1, []),
    "y":       ("y",    1, []),
    "z":       ("z",    1, []),
    "s":       ("s",    1, []),
    "t":       ("t",    1, []),
    "rx":      ("rx",   1, ["theta"]),
    "ry":      ("ry",   1, ["theta"]),
    "rz":      ("rz",   1, ["phi"]),
    "cx":      ("cx",   2, []),
    "cnot":    ("cx",   2, []),
    "cz":      ("cz",   2, []),
    "swap":    ("swap", 2, []),
    "ccx":     ("ccx",  3, []),
    "toffoli": ("ccx",  3, []),
}

_PARAM_DEFAULTS = {"theta": 0.0, "phi": 0.0, "lam": 0.0}

class AerBackend(DVBackend):
    """
    Local DV backend using Qiskit Aer statevector simulation.

    Intended for research experiments — produces exact expectation values
    with zero shot noise. This is the correct DV counterpart to
    StrawberryFieldsBackend for barren plateau comparison studies.

    Uses StatevectorEstimator (qiskit.primitives) which computes ⟨ψ|H|ψ⟩
    exactly from the full statevector, bypassing shot sampling entirely.

    For hardware validation of findings, use QiskitBackend (IBM Runtime).

    Usage:
        backend = AerBackend()
        backend.create_circuit(num_qubits=4)
        backend.add_gate("ry", [0], theta=0.5)
        backend.add_gate("cx", [0, 1])
        backend.execute_circuit()
        energy = backend.compute_expectation(hamiltonian)
    """

    def __init__(self, seed: Optional[int] = None):
        """
        Args:
            seed: Random seed for reproducibility across experiments.
                  Fix this per-experiment to ensure comparable initial
                  conditions between DV and CV runs.
        """
        self.seed = seed
        self._num_qubits: int = 0
        self._operations: List[Dict[str, Any]] = []
        self._circuit: Optional[QuantumCircuit] = None
        self._last_statevector: Optional[np.ndarray] = None
        self._estimator = StatevectorEstimator(seed=seed)
        self._pauli_cache: Optional[List[Tuple[str, complex]]] = None
        self._cached_observable_id: Optional[int] = None

    @property
    def name(self) -> str:
        return f"AerBackend(statevector, seed={self.seed})"

    @property
    def n_qubits(self) -> int:
        return self._num_qubits

    def create_circuit(self, num_qubits: int) -> Dict[str, Any]:
        if num_qubits < 1:
            raise ValueError(f"num_qubits must be >= 1, got {num_qubits}.")
        self._num_qubits = num_qubits
        self._operations = []
        self._circuit = None
        self._last_statevector = None
        return {
            "status": "circuit_created",
            "num_qubits": num_qubits,
            "backend": self.name,
        }

    def add_gate(self, gate_type: str, qubits: List[int], **params) -> Dict[str, Any]:
        if not self._num_qubits:
            raise RuntimeError("Must call create_circuit() before adding gates.")
        gate_key = gate_type.lower()
        if gate_key not in _GATE_DISPATCH:
            raise ValueError(
                f"Gate '{gate_type}' not supported. "
                f"Supported: {sorted(_GATE_DISPATCH.keys())}"
            )
        _, expected_n, _ = _GATE_DISPATCH[gate_key]
        if len(qubits) != expected_n:
            raise ValueError(
                f"Gate '{gate_type}' requires {expected_n} qubit(s), "
                f"got {len(qubits)}."
            )
        if any(q < 0 or q >= self._num_qubits for q in qubits):
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
        Build the QuantumCircuit from the op queue and compute the statevector.
        Statevector is stored internally — available via get_state_vector()
        immediately after this call, before compute_expectation().
        """
        if not self._num_qubits:
            raise RuntimeError("No circuit. Call create_circuit() first.")
        self._circuit = self._build_circuit()
        sv = Statevector(self._circuit)
        self._last_statevector = np.array(sv.data)
        return {
            "status": "completed",
            "backend": self.name,
            "num_qubits": self._num_qubits,
            "num_gates": len(self._operations),
        }

    def compute_expectation(
        self,
        observable: Union[np.ndarray, List[Tuple[str, complex]]]
    ) -> float:
        """
        Compute exact ⟨ψ|H|ψ⟩ via StatevectorEstimator.

        No shot noise. No sampling. The result is the true mathematical
        expectation value of H in state |ψ⟩.

        Args:
            observable: Hermitian np.ndarray of shape (2^n, 2^n), or
                        pre-decomposed Pauli list [(pauli_str, coeff), ...].
                        Pass pre-decomposed terms to skip recomputation
                        across VQE iterations.
        """
        if self._circuit is None:
            raise RuntimeError("No circuit ready. Call execute_circuit() first.")
        pauli_terms = self._get_pauli_terms(observable)
        sparse_op = SparsePauliOp.from_list(
            [(label, coeff) for label, coeff in pauli_terms]
        )
        pub = (self._circuit, sparse_op)
        result = self._estimator.run([pub]).result()
        return float(result[0].data.evs)

    def _get_pauli_terms(
        self,
        observable: Union[np.ndarray, List[Tuple[str, complex]]]
    ) -> List[Tuple[str, complex]]:
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
        Return the exact statevector as complex np.ndarray of shape (2^n,).
        Available after execute_circuit(). No hardware restriction.
        Useful for debugging ansatz structure and validating circuit behavior.
        """
        if self._last_statevector is None:
            raise RuntimeError("No statevector. Call execute_circuit() first.")
        return self._last_statevector.copy()

    def get_probabilities(self) -> np.ndarray:
        """
        Exact bitstring probabilities: |⟨x|ψ⟩|² for all x in {0,1}^n.
        Computed from statevector — no shot noise.
        """
        sv = self.get_state_vector()
        return np.abs(sv) ** 2

    def reset_state(self) -> None:
        """Reset to |0...0⟩. Clears op queue, circuit, and statevector."""
        self._operations = []
        self._circuit = None
        self._last_statevector = None

    def clear_circuit(self) -> None:
        """Clear op queue only. Statevector from last execution remains."""
        self._operations = []
        self._circuit = None

    def _build_circuit(self) -> QuantumCircuit:
        """Construct a QuantumCircuit from the queued operations."""
        qc = QuantumCircuit(self._num_qubits)
        for op in self._operations:
            method_name, _, param_names = _GATE_DISPATCH[op["gate_type"]]
            method = getattr(qc, method_name)
            resolved_params = [
                op["params"].get(p, _PARAM_DEFAULTS.get(p, 0.0))
                for p in param_names
            ]
            method(*resolved_params, *op["qubits"])
        return qc