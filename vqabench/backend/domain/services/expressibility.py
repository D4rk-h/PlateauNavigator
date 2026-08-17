import numpy as np
from backend.domain.models.ansatz import Ansatz
from backend.domain.models.expressibility import CVExpressibilityResult, DVExpressibilityResult, ExpressibilityMethod
from backend.domain.models.hamiltonian import Paradigm
from backend.domain.ports.expressibility_port import ExpressibilityPort


class ExpressibilityService:
    def __init__(self, sampler: ExpressibilityPort, n_bins: int = 75):
        self._sampler = sampler
        self._n_bins = n_bins

    def compute(self, ansatz: Ansatz, n_samples: int = 1000) -> DVExpressibilityResult | CVExpressibilityResult:
        fidelities = self._sampler.sample_fidelities(ansatz, n_samples)
        if ansatz.paradigm == Paradigm.DV:
            return self._compute_dv(ansatz, fidelities)
        return self._compute_cv(ansatz, fidelities)

    def _compute_dv(self, ansatz: Ansatz, fidelities: list[float]) -> DVExpressibilityResult:
        n_qubits = ansatz.n_sites
        hilbert_dim = 2 ** n_qubits

        ansatz_dist = self._build_distribution(fidelities)
        haar_dist = self._haar_distribution_dv(hilbert_dim)
        kl = self._kl_divergence(ansatz_dist, haar_dist)

        return DVExpressibilityResult(
            ansatz_id=ansatz.id,
            method=ExpressibilityMethod.KL_DIVERGENCE,
            score=kl,
            n_samples=len(fidelities),
            n_bins=self._n_bins,
            fidelity_distribution=ansatz_dist.tolist(),
            haar_distribution=haar_dist.tolist(),
            n_qubits=n_qubits,
            hilbert_dim=hilbert_dim
        )

    def _haar_distribution_dv(self, hilbert_dim: int) -> np.ndarray:
        bin_edges = np.linspace(0, 1, self._n_bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        bin_width = 1.0 / self._n_bins
        N = hilbert_dim

        haar = (N - 1) * (1 - bin_centers) ** (N - 2) * bin_width
        haar = np.clip(haar, 1e-10, None)
        return haar / haar.sum()

    def _compute_cv(self, ansatz: Ansatz, fidelities: list[float]) -> CVExpressibilityResult:
        n_modes = ansatz.n_sites
        cutoff = ansatz.fock_cutoff or 10
        hilbert_dim = cutoff ** n_modes

        ansatz_dist = self._build_distribution(fidelities)
        haar_dist = self._haar_distribution_cv(hilbert_dim)
        kl = self._kl_divergence(ansatz_dist, haar_dist)

        return CVExpressibilityResult(
            ansatz_id=ansatz.id,
            method=ExpressibilityMethod.HAAR_FOCK_TRUNCATED,
            score=kl,
            n_samples=len(fidelities),
            n_bins=self._n_bins,
            fidelity_distribution=ansatz_dist.tolist(),
            haar_distribution=haar_dist.tolist(),
            n_modes=n_modes,
            fock_cutoff=cutoff,
            has_non_gaussian=ansatz.has_non_gaussian(),
        )

    def _haar_distribution_cv(self, hilbert_dim: int) -> np.ndarray:
        return self._haar_distribution_dv(hilbert_dim)

    def _build_distribution(self, fidelities: list[float]) -> np.ndarray:
        fidelities_arr = np.clip(np.array(fidelities), 0.0, 1.0)
        counts, _ = np.histogram(fidelities_arr, bins=self._n_bins, range=(0, 1))
        counts = np.clip(counts, 1e-10, None)
        return counts / counts.sum()

    def _kl_divergence(self, p: np.ndarray, q: np.ndarray) -> float:
        p = np.clip(p, 1e-10, None)
        q = np.clip(q, 1e-10, None)
        return float(np.sum(p * np.log(p / q)))

    def compare(self, ansatz_dv: Ansatz, ansatz_cv: Ansatz, n_samples: int = 1000) -> dict:
        result_dv: DVExpressibilityResult = self.compute(ansatz_dv, n_samples)
        result_cv: CVExpressibilityResult = self.compute(ansatz_cv, n_samples)

        return {
            "dv": result_dv.summary(),
            "cv": result_cv.summary(),
            "delta": abs(result_dv.score - result_cv.score),
            "matched": abs(result_dv.score - result_cv.score) < 0.05,
        }