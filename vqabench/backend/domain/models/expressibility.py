from dataclasses import dataclass
from enum import Enum, auto


class ExpressibilityMethod(Enum):
    KL_DIVERGENCE = auto()
    HAAR_FOCK_TRUNCATED = auto()

@dataclass
class ExpressibilityResult:
    ansatz_id: str
    method: ExpressibilityMethod
    score: float
    n_samples: int
    n_bins: int
    fidelity_distribution: list[float]
    haar_distribution: list[float]

    def is_highly_expressible(self, threshold: float = 0.1) -> bool:
        return self.score < threshold

    def summary(self) -> dict:
        return {
            "ansatz_id": self.ansatz_id,
            "method": self.method.name,
            "score": round(self.score, 6),
            "n_samples": self.n_samples,
            "highly_expressive": self.is_highly_expressible(),
        }

@dataclass
class DVExpressibilityResult(ExpressibilityResult):
    n_qubits: int = 0
    hilbert_dim: int = 0

    def summary(self) -> dict:
        return {
            **super().summary(),
            "n_qubits": self.n_qubits,
            "hilbert_dim": self.hilbert_dim,
        }

@dataclass
class CVExpressibilityResult(ExpressibilityResult):
    n_modes: int = 0
    fock_cutoff: int = 0
    has_non_gaussian: bool = False

    @property
    def is_universal(self) -> bool:
        return self.has_non_gaussian

    def summary(self) -> dict:
        return {
            **super().summary(),
            "n_modes": self.n_modes,
            "fock_cutoff": self.fock_cutoff,
            "has_non_gaussian": self.has_non_gaussian,
            "is_universal": self.is_universal,
        }