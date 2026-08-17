from dataclasses import dataclass, field
from typing import Optional, Union
from enum import Enum, auto
from datetime import datetime
import uuid

from backend.domain.models.physical_model import PhysicalModel
from backend.domain.models.hamiltonian import Paradigm
from backend.domain.models.ansatz import Ansatz
from backend.domain.models.job import Job, JobStatus
from backend.domain.models.expressibility import (
    DVExpressibilityResult,
    CVExpressibilityResult,
)


class ExperimentStatus(Enum):
    CREATED = auto()
    RUNNING = auto()
    COMPLETED = auto()
    FAILED = auto()

@dataclass
class SingleExperiment:
    name: str
    model: PhysicalModel
    ansatz: Ansatz
    job: Job
    expressibility: Optional[
        Union[DVExpressibilityResult, CVExpressibilityResult]
    ] = None
    notes: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def __post_init__(self):
        if self.model.n_sites != self.ansatz.n_sites:
            raise ValueError(
                f"Model n_sites {self.model.n_sites} does not match "
                f"Ansatz n_sites {self.ansatz.n_sites}"
            )
        if self.ansatz.paradigm != self.job.circuit.paradigm:
            raise ValueError(
                f"Ansatz paradigm {self.ansatz.paradigm} does not match "
                f"Job circuit paradigm {self.job.circuit.paradigm}"
            )

    @property
    def status(self) -> ExperimentStatus:
        mapping = {
            JobStatus.PENDING: ExperimentStatus.CREATED,
            JobStatus.RUNNING: ExperimentStatus.RUNNING,
            JobStatus.COMPLETED: ExperimentStatus.COMPLETED,
            JobStatus.FAILED: ExperimentStatus.FAILED,
            JobStatus.CANCELLED: ExperimentStatus.FAILED,
        }
        return mapping[self.job.status]

    @property
    def paradigm(self) -> Paradigm:
        return self.ansatz.paradigm

    def is_complete(self) -> bool:
        return self.status == ExperimentStatus.COMPLETED

    def has_expressibility(self) -> bool:
        return self.expressibility is not None

    def final_energy(self) -> Optional[float]:
        if self.job.result:
            return self.job.result.final_energy
        return None

    def best_energy(self) -> Optional[float]:
        if self.job.result:
            return self.job.result.best_energy()
        return None

    def expressibility_score(self) -> Optional[float]:
        if self.expressibility:
            return self.expressibility.score
        return None

    def summary(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "status": self.status.name,
            "paradigm": self.paradigm.name,
            "model": self.model.__class__.__name__,
            "n_sites": self.model.n_sites,
            "ansatz": self.ansatz.name,
            "ansatz_type": self.ansatz.ansatz_type.name,
            "final_energy": self.final_energy(),
            "best_energy": self.best_energy(),
            "expressibility_score": self.expressibility_score(),
            "created_at": self.created_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }

@dataclass
class ComparisonResult:
    dv_final_energy: float
    cv_final_energy: float
    dv_best_energy: float
    cv_best_energy: float
    dv_iterations: int
    cv_iterations: int
    dv_expressibility_score: Optional[float]
    cv_expressibility_score: Optional[float]
    expressibility_delta: Optional[float]
    created_at: datetime = field(default_factory=datetime.now)

    def energy_delta(self) -> float:
        return abs(self.dv_final_energy - self.cv_final_energy)

    def best_energy_delta(self) -> float:
        return abs(self.dv_best_energy - self.cv_best_energy)

    def winner_by_energy(self) -> str:
        if self.dv_final_energy < self.cv_final_energy:
            return "DV"
        elif self.cv_final_energy < self.dv_final_energy:
            return "CV"
        else:
            return "TIE"

    def winner_by_best_energy(self) -> str:
        if self.dv_best_energy < self.cv_best_energy:
            return "DV"
        elif self.cv_best_energy < self.dv_best_energy:
            return "CV"
        else:
            return "TIE"

    def expressibility_matched(self, threshold: float = 0.05) -> bool:
        if self.expressibility_delta is None:
            return False
        return self.expressibility_delta < threshold

    def summary(self) -> dict:
        return {
            "dv_final_energy": self.dv_final_energy,
            "cv_final_energy": self.cv_final_energy,
            "dv_best_energy": self.dv_best_energy,
            "cv_best_energy": self.cv_best_energy,
            "energy_delta": self.energy_delta(),
            "best_energy_delta": self.best_energy_delta(),
            "winner_by_energy": self.winner_by_energy(),
            "winner_by_best_energy": self.winner_by_best_energy(),
            "dv_iterations": self.dv_iterations,
            "cv_iterations": self.cv_iterations,
            "dv_expressibility_score": self.dv_expressibility_score,
            "cv_expressibility_score": self.cv_expressibility_score,
            "expressibility_delta": self.expressibility_delta,
            "expressibility_matched": self.expressibility_matched(),
        }

@dataclass
class ComparisonExperiment:
    name: str
    model: PhysicalModel
    dv_experiment: SingleExperiment
    cv_experiment: SingleExperiment
    comparison_result: Optional[ComparisonResult] = None
    notes: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def __post_init__(self):
        if self.dv_experiment.paradigm != Paradigm.DV:
            raise ValueError(
                f"dv_experiment must use DV paradigm, got {self.dv_experiment.paradigm}"
            )
        if self.cv_experiment.paradigm != Paradigm.CV:
            raise ValueError(
                f"cv_experiment must use CV paradigm, got {self.cv_experiment.paradigm}"
            )
        if self.dv_experiment.model.n_sites != self.cv_experiment.model.n_sites:
            raise ValueError(
                f"DV and CV experiments must have the same n_sites. DV: {self.dv_experiment.model.n_sites}, CV: {self.cv_experiment.model.n_sites}"
            )

    @property
    def status(self) -> ExperimentStatus:
        dv = self.dv_experiment.status
        cv = self.cv_experiment.status

        if ExperimentStatus.FAILED in {dv, cv}:
            return ExperimentStatus.FAILED
        if dv == ExperimentStatus.COMPLETED and cv == ExperimentStatus.COMPLETED:
            return ExperimentStatus.COMPLETED
        if ExperimentStatus.RUNNING in {dv, cv}:
            return ExperimentStatus.RUNNING
        return ExperimentStatus.CREATED

    def is_complete(self) -> bool:
        return self.status == ExperimentStatus.COMPLETED

    def has_comparison_result(self) -> bool:
        return self.comparison_result is not None

    def expressibility_matched(self, threshold: float = 0.05) -> bool:
        if self.comparison_result is None:
            return False
        return self.comparison_result.expressibility_matched(threshold)

    def build_comparison_result(self) -> ComparisonResult:
        if not self.dv_experiment.is_complete():
            raise ValueError("DV experiment is not complete.")
        if not self.cv_experiment.is_complete():
            raise ValueError("CV experiment is not complete.")

        dv_result = self.dv_experiment.job.result
        cv_result = self.cv_experiment.job.result

        dv_expr = self.dv_experiment.expressibility_score()
        cv_expr = self.cv_experiment.expressibility_score()
        expr_delta = (
            abs(dv_expr - cv_expr)
            if dv_expr is not None and cv_expr is not None
            else None
        )
        self.comparison_result = ComparisonResult(
            dv_final_energy=dv_result.final_energy,
            cv_final_energy=cv_result.final_energy,
            dv_best_energy=dv_result.best_energy(),
            cv_best_energy=cv_result.best_energy(),
            dv_iterations=dv_result.iterations,
            cv_iterations=cv_result.iterations,
            dv_expressibility_score=dv_expr,
            cv_expressibility_score=cv_expr,
            expressibility_delta=expr_delta,
        )
        return self.comparison_result

    def summary(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "status": self.status.name,
            "model": self.model.__class__.__name__,
            "n_sites": self.model.n_sites,
            "dv_experiment": self.dv_experiment.summary(),
            "cv_experiment": self.cv_experiment.summary(),
            "comparison_result": (
                self.comparison_result.summary() if self.comparison_result else None
            ),
            "created_at": self.created_at.isoformat(),
        }