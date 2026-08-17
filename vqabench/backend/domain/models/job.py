from dataclasses import dataclass, field
from typing import List, Optional
from enum import Enum, auto
import uuid
from datetime import datetime

from backend.domain.models.parameter import Parameter
from backend.domain.models.hamiltonian import Hamiltonian
from backend.domain.models.circuit import Circuit


class JobStatus(Enum):
    PENDING = auto()
    RUNNING = auto()
    COMPLETED = auto()
    FAILED = auto()
    CANCELLED = auto()


class OptimizerType(Enum):
    SPSA = auto()
    ADAM = auto()
    L_BFGS_B = auto()
    COBYLA = auto()
    GRADIENT_DESCENT = auto()
    CUSTOM = auto()

@dataclass
class JobConfig:
    optimizer: OptimizerType
    max_iterations: int
    learning_rate: Optional[float] = None
    shots: Optional[int] = None
    random_seed: Optional[int] = None
    convergence_tolerance: Optional[float] = None
    gradient_samples: Optional[int] = None

    def __post_init__(self):
        if self.max_iterations <= 0:
            raise ValueError("max_iterations must be positive")
        if self.shots is not None and self.shots <= 0:
            raise ValueError("shots must be positive")
        if self.learning_rate is not None and self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.convergence_tolerance is not None and self.convergence_tolerance <= 0:
            raise ValueError("convergence_tolerance must be positive")
        if self.gradient_samples is not None and self.gradient_samples <= 0:
            raise ValueError("gradient_samples must be positive")

    def is_gradient_free(self) -> bool:
        return self.optimizer in {OptimizerType.COBYLA, OptimizerType.SPSA}
    
@dataclass
class JobResult:
    final_energy: float
    final_parameters: List[Parameter] = field(default_factory=list)
    energy_history: List[float] = field(default_factory=list)
    gradient_norm_history: List[float] = field(default_factory=list)
    iterations: int = 0
    function_evaluations: int = 0
    converged: bool = False
    execution_time_seconds: float = 0.0
    error_message: Optional[str] = None

    def __post_init__(self):
        if self.iterations < 0:
            raise ValueError("iterations must be non-negative")
        if self.execution_time_seconds < 0:
            raise ValueError("execution_time_seconds must be non-negative")
        if self.function_evaluations < 0:
            raise ValueError("function_evaluations must be non-negative")

    def best_energy(self) -> float:
        if self.energy_history:
            return min(self.energy_history)
        return self.final_energy

    def convergence_rate(self) -> Optional[float]:
        if len(self.energy_history) < 2:
            return None
        total_drop = self.energy_history[0] - self.energy_history[-1]
        return total_drop / len(self.energy_history)

    def has_error(self) -> bool:
        return self.error_message is not None

    def summary(self) -> dict:
        return {
            "final_energy": self.final_energy,
            "best_energy": self.best_energy(),
            "iterations": self.iterations,
            "function_evaluations": self.function_evaluations,
            "converged": self.converged,
            "execution_time_seconds": self.execution_time_seconds,
            "convergence_rate": self.convergence_rate(),
            "has_error": self.has_error(),
        }
    
@dataclass
class Job:
    name: str
    hamiltonian: Hamiltonian
    circuit: Circuit
    config: JobConfig
    status: JobStatus = JobStatus.PENDING
    result: Optional[JobResult] = None
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def __post_init__(self):
        if self.hamiltonian.paradigm != self.circuit.paradigm:
            raise ValueError(
                f"Hamiltonian paradigm {self.hamiltonian.paradigm} "
                f"does not match Circuit paradigm {self.circuit.paradigm}"
            )
        if self.hamiltonian.n_sites != self.circuit.n_sites:
            raise ValueError(
                f"Hamiltonian n_sites {self.hamiltonian.n_sites} "
                f"does not match Circuit n_sites {self.circuit.n_sites}"
            )
        if self.result is not None and self.status == JobStatus.PENDING:
            raise ValueError("Job with result cannot be in PENDING status")
        if self.completed_at is not None and self.started_at is None:
            raise ValueError("completed_at requires started_at")

    def mark_running(self) -> None:
        if self.status != JobStatus.PENDING:
            raise ValueError(f"Cannot start a job in {self.status.name} status. Job must be PENDING.")
        self.status = JobStatus.RUNNING
        self.started_at = datetime.now()

    def mark_completed(self, result: JobResult) -> None:
        if self.status != JobStatus.RUNNING:
            raise ValueError(f"Cannot complete a job in {self.status.name} status. Job must be RUNNING.")
        self.status = JobStatus.COMPLETED
        self.result = result
        self.completed_at = datetime.now()

    def mark_failed(self, error: str) -> None:
        if self.status not in {JobStatus.RUNNING, JobStatus.PENDING}:
            raise ValueError(f"Cannot fail a job in {self.status.name} status. Job must be RUNNING or PENDING.")
        self.status = JobStatus.FAILED
        self.completed_at = datetime.now()
        if self.result is None:
            self.result = JobResult(
                final_energy=float('inf'),
                error_message=error,
            )

    def mark_cancelled(self) -> None:
        if self.status not in {JobStatus.RUNNING, JobStatus.PENDING}:
            raise ValueError(f"Cannot cancel a job in {self.status.name} status. Job must be RUNNING or PENDING.")
        self.status = JobStatus.CANCELLED
        self.completed_at = datetime.now()

    def is_ready_to_run(self) -> bool:
        return self.status == JobStatus.PENDING

    def is_terminal(self) -> bool:
        return self.status in {JobStatus.COMPLETED, JobStatus.FAILED, JobStatus.CANCELLED}

    def duration_seconds(self) -> Optional[float]:
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None

    def summary(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "status": self.status.name,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "duration_seconds": self.duration_seconds(),
            "result": self.result.summary() if self.result else None,
        }