import numpy as np
from typing import Tuple, List
import itertools

_PAULIS = {
    "I": np.eye(2, dtype=complex),
    "X": np.array([[0, 1], [1, 0]], dtype=complex),
    "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
    "Z": np.array([[1, 0], [0, -1]], dtype=complex),
}

def _pauli_decompose(H: np.ndarray) -> List[Tuple[str, complex]]:
    """
    Decompose a Hermitian matrix H into the Pauli basis.

    H = Σᵢ cᵢ Pᵢ  where  cᵢ = Tr(Pᵢ H) / 2^n

    Args:
        H: Hermitian np.ndarray, shape (2^n, 2^n).
    Returns:
        List of (pauli_string, coefficient) tuples, zero terms excluded.
        e.g. [('IZ', 0.5+0j), ('XX', -0.3+0j)]
    """
    dim = H.shape[0]
    n = int(np.log2(dim))
    if 2**n != dim:
        raise ValueError(
            f"Observable dimension {dim} is not a power of 2."
        )
    norm = 1.0 / dim
    terms = []
    for labels in itertools.product("IXYZ", repeat=n):
        label = "".join(labels)
        P = _PAULIS[labels[0]]
        for l in labels[1:]:
            P = np.kron(P, _PAULIS[l])
        coeff = np.trace(P @ H) * norm
        if abs(coeff) > 1e-12:
            terms.append((label, coeff))
    return terms
