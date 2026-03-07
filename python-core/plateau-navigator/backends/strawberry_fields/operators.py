import numpy as np


def _annihilation_op(d: int) -> np.ndarray:
    """â in Fock basis: â|n⟩ = √n |n-1⟩"""
    return np.diag(np.sqrt(np.arange(1, d, dtype=complex)), k=1)

def _creation_op(d: int) -> np.ndarray:
    """â† in Fock basis: â†|n⟩ = √(n+1) |n+1⟩"""
    return np.diag(np.sqrt(np.arange(1, d, dtype=complex)), k=-1)

def _number_op(d: int) -> np.ndarray:
    """n̂ = â†â"""
    return np.diag(np.arange(d, dtype=complex))

def _quadrature_x(d: int) -> np.ndarray:
    """X̂ = (â + â†) / √2"""
    a = _annihilation_op(d)
    return (a + a.conj().T) / np.sqrt(2)

def _quadrature_p(d: int) -> np.ndarray:
    """P̂ = -i(â - â†) / √2"""
    a = _annihilation_op(d)
    return -1j * (a - a.conj().T) / np.sqrt(2)
