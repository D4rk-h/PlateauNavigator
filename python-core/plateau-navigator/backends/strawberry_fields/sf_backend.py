from ..backend_interface import CVBackend
from typing import Dict, Any, List, Optional, Union
import numpy as np
import strawberryfields as sf
from strawberryfields import ops
import operators as op


# Maps gate_type string -> (SF ops class, required_modes, required_params)
_GATE_REGISTRY: Dict[str, Dict[str, Any]] = {
    "sgate":               {"op": ops.Sgate,  "n_modes": 1, "params": ["r", "phi"]},
    "squeezing":           {"op": ops.Sgate,  "n_modes": 1, "params": ["r", "phi"]},
    "dgate":               {"op": ops.Dgate,  "n_modes": 1, "params": ["r", "phi"]},
    "displacement":        {"op": ops.Dgate,  "n_modes": 1, "params": ["r", "phi"]},
    "rgate":               {"op": ops.Rgate,  "n_modes": 1, "params": ["phi"]},
    "rotation":            {"op": ops.Rgate,  "n_modes": 1, "params": ["phi"]},
    "kgate":               {"op": ops.Kgate,  "n_modes": 1, "params": ["kappa"],  "non_gaussian": True},
    "kerr":                {"op": ops.Kgate,  "n_modes": 1, "params": ["kappa"],  "non_gaussian": True},
    "vgate":               {"op": ops.Vgate,  "n_modes": 1, "params": ["gamma"],  "non_gaussian": True},
    "cubic_phase":         {"op": ops.Vgate,  "n_modes": 1, "params": ["gamma"],  "non_gaussian": True},
    "bsgate":              {"op": ops.BSgate, "n_modes": 2, "params": ["theta", "phi"]},
    "beamsplitter":        {"op": ops.BSgate, "n_modes": 2, "params": ["theta", "phi"]},
    "s2gate":              {"op": ops.S2gate, "n_modes": 2, "params": ["r", "phi"]},
    "two_mode_squeezing":  {"op": ops.S2gate, "n_modes": 2, "params": ["r", "phi"]},
    "mzgate":              {"op": ops.MZgate, "n_modes": 2, "params": ["phi_in", "phi_ex"]},
    "mach_zehnder":        {"op": ops.MZgate, "n_modes": 2, "params": ["phi_in", "phi_ex"]},
}

_PARAM_DEFAULTS = {
    "r": 0.0, "phi": 0.0, "theta": 0.0, "kappa": 0.0, "gamma": 0.0, "phi_in": 0.0, "phi_ex": 0.0,
}

_STRING_OBSERVABLES = {
    "x":  op._quadrature_x,
    "p":  op._quadrature_p,
    "n":  op._number_op,
    "x2": lambda d: op._quadrature_x(d) @ op._quadrature_x(d),
    "p2": lambda d: op._quadrature_p(d) @ op._quadrature_p(d),
}


class StrawberryFieldsBackend(CVBackend):
    """
    Continuous-Variable (CV) backend using Strawberry Fields.

    Supports three SF engine backends:
        'fock'     — general Fock-basis simulation (truncated at cutoff_dim)
        'gaussian' — efficient exact simulation for Gaussian states only;
                     non-Gaussian gates (kgate, vgate) will raise.
        'tf'       — TensorFlow backend; enables backprop-based gradient
                     computation as an alternative to parameter shift.
                
    Usage:
        backend = StrawberryFieldsBackend(backend_type="fock", cutoff_dim=10)
        backend.create_circuit(n_modes=2)
        backend.apply_op("sgate", [0], r=0.5)
        backend.apply_op("bsgate", [0, 1], theta=np.pi/4)
        backend.execute_circuit()
        H = StrawberryFieldsBackend.n_op(10)
        energy = backend.compute_expectation(H, mode=0)
    """

    @staticmethod
    def a(cutoff_dim: int) -> np.ndarray:
        return op._annihilation_op(cutoff_dim)

    @staticmethod
    def adag(cutoff_dim: int) -> np.ndarray:
        return op._creation_op(cutoff_dim)

    @staticmethod
    def n_op(cutoff_dim: int) -> np.ndarray:
        return op._number_op(cutoff_dim)

    @staticmethod
    def x_op(cutoff_dim: int) -> np.ndarray:
        return op._quadrature_x(cutoff_dim)

    @staticmethod
    def p_op(cutoff_dim: int) -> np.ndarray:
        return op._quadrature_p(cutoff_dim)

    def __init__(self, backend_type: str = "fock", cutoff_dim: int = 6):
        if backend_type not in ("fock", "gaussian", "tf"):
            raise ValueError(
                f"Unknown backend_type '{backend_type}'. "
                f"Must be one of: 'fock', 'gaussian', 'tf'."
            )
        self._backend_type = backend_type
        self._cutoff_dim = cutoff_dim
        self._num_modes: Optional[int] = None
        self._operations: List[Dict[str, Any]] = []
        self._last_state = None
        self._engine: Optional[sf.Engine] = None

    @property
    def name(self) -> str:
        return f"StrawberryFields-{self._backend_type}(cutoff={self._cutoff_dim})"

    @property
    def n_modes(self) -> int:
        if self._num_modes is None:
            raise RuntimeError("Circuit not yet created. Call create_circuit() first.")
        return self._num_modes

    @property
    def backend_type(self) -> str:
        return self._backend_type

    def create_circuit(self, n_modes: int, cutoff_dim: int = None) -> Dict[str, Any]:
        """
        Initialize a CV circuit.

        Args:
            n_modes:    Number of bosonic modes.
            cutoff_dim: Override instance cutoff_dim for this circuit.
                        Useful for systematic truncation studies.
        """
        if n_modes < 1:
            raise ValueError(f"n_modes must be >= 1, got {n_modes}.")
        self._num_modes = n_modes
        if cutoff_dim is not None:
            self._cutoff_dim = cutoff_dim
        self._operations = []
        self._last_state = None

        # Rebuild engine when circuit topology changes
        backend_options = {}
        if self._backend_type in ("fock", "tf"):
            backend_options["cutoff_dim"] = self._cutoff_dim
        self._engine = sf.Engine(self._backend_type, backend_options=backend_options)
        return {
            "status": "circuit_created",
            "n_modes": n_modes,
            "cutoff_dim": self._cutoff_dim,
            "backend": self.name,
        }

    def apply_op(self, op_type: str, modes: List[int], **params) -> Dict[str, Any]:
        """
        Queue a CV operation. See class docstring for gate types and params.

        Raises:
            ValueError:        Unknown gate, wrong number of modes, or
                               non-Gaussian gate on a Gaussian backend.
            RuntimeError:      Called before create_circuit().
            IndexError:        Mode index out of range.
        """
        if self._num_modes is None:
            raise RuntimeError("Must call create_circuit() before applying operations.")

        gate_key = op_type.lower()
        if gate_key not in _GATE_REGISTRY:
            raise ValueError(
                f"Unknown gate '{op_type}'. "
                f"Supported: {sorted(_GATE_REGISTRY.keys())}"
            )

        spec = _GATE_REGISTRY[gate_key]

        if len(modes) != spec["n_modes"]:
            raise ValueError(
                f"Gate '{op_type}' requires {spec['n_modes']} mode(s), "
                f"got {len(modes)}: {modes}."
            )

        if any(m >= self._num_modes or m < 0 for m in modes):
            raise IndexError(
                f"Mode indices {modes} out of range for circuit with "
                f"{self._num_modes} modes (0-indexed)."
            )

        if spec.get("non_gaussian") and self._backend_type == "gaussian":
            raise ValueError(
                f"Gate '{op_type}' is non-Gaussian and cannot run on the "
                f"'gaussian' backend. Switch to 'fock' or 'tf'."
            )

        self._operations.append({
            "gate_type": gate_key,
            "modes": modes,
            "params": params,
        })
        return {"status": "op_queued", "gate_type": op_type, "modes": modes}

    def execute_circuit(self) -> Dict[str, Any]:
        """Build and run the sf.Program. Reuses the existing sf.Engine."""
        if self._num_modes is None:
            raise RuntimeError("No circuit found. Call create_circuit() first.")
        if self._engine is None:
            raise RuntimeError("Engine not initialized. Call create_circuit() first.")
        prog = sf.Program(self._num_modes)
        with prog.context as q:
            for op in self._operations:
                self._dispatch_gate(op, q)
        result = self._engine.run(prog, reset=True)
        self._last_state = result.state

        return {
            "status": "completed",
            "backend": self.name,
            "n_modes": self._num_modes,
            "n_ops": len(self._operations),
        }

    def get_state(self) -> Any:
        """
        Return the post-execution state in its native representation.

        'fock' / 'tf' → complex np.ndarray, shape (cutoff_dim,) * n_modes
        'gaussian'    → Tuple[np.ndarray, np.ndarray]: (mu, cov)
                        mu  shape: (2 * n_modes,)
                        cov shape: (2 * n_modes, 2 * n_modes)
        """
        self._require_state()
        if self._backend_type == "gaussian":
            mu = self._last_state.means()
            cov = self._last_state.cov()
            return mu, cov
        return np.array(self._last_state.ket())

    def get_fock_probabilities(self, cutoff: int = None) -> np.ndarray:
        """
        Photon-number probability tensor.

        Shape: (cutoff_dim,) * n_modes
        Entry [n0, n1, ...] = P(mode0 has n0 photons AND mode1 has n1 photons ...)

        The joint structure is preserved — do NOT flatten this for analysis.
        Marginal distributions: probs.sum(axis=1) gives mode-0 marginal, etc.
        """
        self._require_state()
        if self._backend_type == "gaussian":
            d = cutoff or self._cutoff_dim
            return np.array(self._last_state.fock_prob(
                list(range(d)) * self._num_modes
            ))
        ket = np.array(self._last_state.ket())
        return np.abs(ket) ** 2

    def measure_homodyne(self, phi: float, mode: int) -> float:
        self._require_state()
        return float(self._last_state.quad_expectation(mode, phi=phi)[0])

    def measure_heterodyne(self, mode: int) -> complex:
        self._require_state()
        xval = self._last_state.quad_expectation(mode, phi=0.0)[0]
        pval = self._last_state.quad_expectation(mode, phi=np.pi / 2)[0]
        return complex(xval, pval) / np.sqrt(2)

    # ── Expectation values ────────────────────────────────────────────

    def compute_expectation(
        self,
        observable: Union[str, np.ndarray],
        mode: int = 0,
    ) -> float:
        """
        Compute ⟨ψ|O|ψ⟩.
        Args:
            observable: String shortcut ('x', 'p', 'n', 'x2', 'p2') or
                        Hermitian np.ndarray of shape (cutoff_dim, cutoff_dim).
            mode:       Target mode index (for single-mode observables on
                        multi-mode states).
        Raises:
            ValueError: Observable string unknown, or matrix dimension mismatch.
            TypeError:  Observable is neither str nor np.ndarray.
        """
        self._require_state()
        H = self._resolve_observable(observable)
        self._validate_observable(H)
        ket = np.array(self._last_state.ket())
        if ket.ndim == 1:
            return self._expectation_single_mode(H, ket)
        else:
            return self._expectation_reduced(H, ket, mode)

    def _resolve_observable(self, observable: Union[str, np.ndarray]) -> np.ndarray:
        if isinstance(observable, str):
            key = observable.lower()
            if key not in _STRING_OBSERVABLES:
                raise ValueError(
                    f"Unknown observable '{observable}'. "
                    f"Supported strings: {list(_STRING_OBSERVABLES.keys())}"
                )
            return _STRING_OBSERVABLES[key](self._cutoff_dim)
        if isinstance(observable, np.ndarray):
            return observable
        raise TypeError(
            f"observable must be a string or np.ndarray, got {type(observable)}."
        )

    def _validate_observable(self, H: np.ndarray) -> None:
        if H.ndim != 2 or H.shape[0] != H.shape[1]:
            raise ValueError(
                f"Observable must be a square 2D matrix, got shape {H.shape}."
            )
        if H.shape[0] != self._cutoff_dim:
            raise ValueError(
                f"Observable dimension {H.shape[0]} does not match "
                f"cutoff_dim {self._cutoff_dim}. Rebuild observable with "
                f"the correct cutoff or adjust cutoff_dim."
            )
        if not np.allclose(H, H.conj().T, atol=1e-10):
            raise ValueError("Observable must be Hermitian (H = H†).")

    def _expectation_single_mode(self, H: np.ndarray, ket: np.ndarray) -> float:
        return float(np.real(ket.conj() @ H @ ket))

    def _expectation_reduced(self, H: np.ndarray, ket: np.ndarray, mode: int) -> float:
        """Compute Tr(H * ρ_mode) where ρ_mode is the reduced density matrix."""
        d = self._cutoff_dim
        n_modes = ket.ndim
        if mode < 0 or mode >= n_modes:
            raise IndexError(
                f"mode {mode} out of range for {n_modes}-mode state."
            )
        axes = [mode] + [i for i in range(n_modes) if i != mode]
        ket_t = np.transpose(ket, axes).reshape(d, -1)
        rho_reduced = ket_t @ ket_t.conj().T
        return float(np.real(np.trace(H @ rho_reduced)))

    def reset_state(self) -> None:
        """Reset to vacuum: clear op queue and last state, keep circuit config."""
        self._operations = []
        self._last_state = None

    def clear_circuit(self) -> None:
        """Clear op queue only. State from last execution remains accessible."""
        self._operations = []

    def mean_photon_per_mode(self) -> List[float]:
        """Return ⟨n̂⟩ for each mode. Convenience wrapper around SF's state API."""
        self._require_state()
        return [
            float(self._last_state.mean_photon(m)[0])
            for m in range(self._num_modes)
        ]

    def _require_state(self) -> None:
        if self._last_state is None:
            raise RuntimeError(
                "No state available. Call execute_circuit() first."
            )

    def _dispatch_gate(self, op: Dict[str, Any], q) -> None:
        """Apply a single queued operation inside an active sf.Program context."""
        spec = _GATE_REGISTRY[op["gate_type"]]
        sf_op_class = spec["op"]
        param_names = spec["params"]
        modes = op["modes"]
        provided = op["params"]

        # Resolve params in declaration order, falling back to defaults
        resolved = [
            provided.get(pname, _PARAM_DEFAULTS.get(pname, 0.0))
            for pname in param_names
        ]

        mode_args = tuple(q[m] for m in modes)
        sf_op_class(*resolved) | mode_args