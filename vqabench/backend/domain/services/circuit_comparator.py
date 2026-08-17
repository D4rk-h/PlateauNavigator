from backend.domain.models.ansatz import Ansatz
from backend.domain.models.expressibility import (
    DVExpressibilityResult,
    CVExpressibilityResult,
)
from backend.domain.services.expressibility import ExpressibilityService


class CircuitComparatorService:
    def __init__(
        self,
        expressibility_service: ExpressibilityService,
        threshold: float = 0.05,
        n_samples: int = 1000,
    ):
        self._service = expressibility_service
        self._threshold = threshold
        self._n_samples = n_samples

    def are_comparable(self, ansatz_dv: Ansatz, ansatz_cv: Ansatz) -> bool:
        result = self.compare(ansatz_dv, ansatz_cv)
        return result["matched"]

    def compare(self, ansatz_dv: Ansatz, ansatz_cv: Ansatz) -> dict:

        result_dv: DVExpressibilityResult = self._service.compute(
            ansatz_dv, n_samples=self._n_samples
        )
        result_cv: CVExpressibilityResult = self._service.compute(
            ansatz_cv, n_samples=self._n_samples
        )

        delta = abs(result_dv.score - result_cv.score)
        matched = delta < self._threshold

        return {
            "dv": result_dv.summary(),
            "cv": result_cv.summary(),
            "delta": delta,
            "matched": matched,
            "warning": (
                None
                if matched
                else f"Expressibility delta {delta:.4f} exceeds threshold {self._threshold:.4f}. Comparison may not be meaningful."
            ),
        }

    def find_matching_depth(
            self,
            ansatz_dv: Ansatz,
            target_score: float,
            max_layers: int = 10,
    ) -> int:
        from dataclasses import replace

        best_layers = 1
        best_delta = float("inf")

        for n_layers in range(1, max_layers + 1):
            candidate = Ansatz(
                name=ansatz_dv.name,
                paradigm=ansatz_dv.paradigm,
                n_sites=ansatz_dv.n_sites,
                ansatz_type=ansatz_dv.ansatz_type,
                n_layers=n_layers,
                entanglement_pattern=ansatz_dv.entanglement_pattern,
            )
            result = self._service.compute(candidate, self._n_samples)
            delta = abs(result.score - target_score)

            if delta < best_delta:
                best_delta = delta
                best_layers = n_layers

        return best_layers