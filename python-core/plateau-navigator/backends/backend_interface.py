from abc import ABC, abstractmethod
from typing import Dict, Any, List, Union
import numpy as np



class DVBackend(ABC):
    """
    Abstract base class for Discrete Variable (DV) quantum backends.

    DV quantum computation operates on qubits, with finite-dimensional 
    Hilbert spaces.
    """

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


class CVBackend(ABC):
    """
    Abstract base class for Continuous Variable (CV) quantum backends.

    CV quantum computation operates on bosonic modes (optical or microwave),
    with infinite-dimensional Hilbert spaces truncated at cutoff_dim in Fock
    representations, or represented exactly via Gaussian covariance matrices.

    Supported backend engines: Strawberry Fields (fock/gaussian/bosonic/tf)
    """

    @abstractmethod
    def create_circuit(self, n_modes: int, cutoff_dim: int = 10) -> Dict[str, Any]:
        """
        Initialize a CV circuit.

        Args:
            n_modes:    Number of bosonic modes.
            cutoff_dim: Fock space truncation dimension. Only relevant for
                        non-Gaussian (Fock) backends. Higher = more accurate
                        but exponentially more memory: O(cutoff_dim^n_modes).
                        Ignored by Gaussian backends.
        """
        pass

    @abstractmethod
    def apply_op(self, op_type: str, modes: List[int], **params) -> Dict[str, Any]:
        """
        Apply a CV gate/operation to specified modes.

        Supported op_type values and their required params:
            'D'  - Displacement:   alpha (complex)
            'S'  - Squeezing:      r (float), phi (float)
            'R'  - Rotation:       theta (float)
            'BS' - Beamsplitter:   theta (float), phi (float)  [2 modes]
            'S2' - Two-mode squeezing: r (float), phi (float)  [2 modes]
            'K'  - Kerr:           kappa (float)               [non-Gaussian]
            'V'  - Cubic phase:    gamma (float)               [non-Gaussian]
            'MZ' - Mach-Zehnder:   phi_in (float), phi_ex (float) [2 modes]

        Non-Gaussian ops (K, V) require a Fock backend — will raise
        NotImplementedError on Gaussian backends.
        """
        pass

    @abstractmethod
    def execute_circuit(self) -> Dict[str, Any]:
        pass

    @abstractmethod
    def get_state(self) -> Any:
        """
        Return the post-execution state in the backend's native representation.

        Return type depends on backend_type:
            'fock':     np.ndarray of shape (cutoff_dim,) * n_modes, dtype=complex
            'gaussian': Tuple[np.ndarray, np.ndarray] -> (mu, cov)
                        mu:  displacement vector, shape (2 * n_modes,)
                        cov: covariance matrix,   shape (2 * n_modes, 2 * n_modes)

        Use backend_type property to handle the return correctly upstream.
        """
        pass

    @abstractmethod
    def get_fock_probabilities(self, cutoff: int = None) -> np.ndarray:
        """
        Photon-number probability distribution in the Fock basis.

        Returns:
            np.ndarray of shape (cutoff_dim,) * n_modes where entry [n1, n2, ...]
            is the probability of measuring n1 photons in mode 1, n2 in mode 2, etc.

        Args:
            cutoff: Override truncation for this call. None uses circuit cutoff_dim.

        Note: For Gaussian backends this is computed via the Hafnian and can be
        expensive. For Fock backends it's |amplitude|^2 directly.
        """
        pass

    @abstractmethod
    def measure_homodyne(self, phi: float, mode: int) -> float:
        """
        Homodyne measurement of quadrature X*cos(phi) + P*sin(phi).

        phi=0   -> X quadrature (position)
        phi=π/2 -> P quadrature (momentum)

        Returns a continuous real-valued outcome. Collapses the state.
        """
        pass

    @abstractmethod
    def measure_heterodyne(self, mode: int) -> complex:
        """
        Heterodyne measurement. Projects onto coherent states.
        Returns complex amplitude alpha = (x + ip) / sqrt(2).
        Collapses the state.
        """
        pass

    @abstractmethod
    def compute_expectation(
        self,
        observable: Union[str, np.ndarray],
        mode: int = 0
    ) -> float:
        """
        Compute <ψ|O|ψ> for a given observable.

        Args:
            observable: One of:
                - 'x'  : position quadrature X̂ = (â + â†) / sqrt(2)
                - 'p'  : momentum quadrature P̂ = -i(â - â†) / sqrt(2)
                - 'n'  : photon number n̂ = â†â
                - 'x2' : X̂²  (variance-related)
                - 'p2' : P̂²
                - np.ndarray: arbitrary Hermitian matrix in Fock basis,
                              shape (cutoff_dim, cutoff_dim) for single mode
            mode: Which mode to measure (for string observables).

        Returns:
            Real-valued expectation value.
        """
        pass

    @abstractmethod
    def reset_state(self) -> None:
        """Reset all modes to vacuum |0⟩."""
        pass

    @abstractmethod
    def clear_circuit(self) -> None:
        """Clear the operation queue without resetting state."""
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    @abstractmethod
    def n_modes(self) -> int:
        """Number of modes in the current circuit."""
        pass

    @property
    @abstractmethod
    def backend_type(self) -> str:
        """
        Engine type string. Must be one of: 'fock', 'gaussian', 'bosonic', 'tf'.
        Used upstream (VQE, analysis layer) to correctly interpret get_state()
        and to validate that applied operations are compatible with the engine.
        """
        pass