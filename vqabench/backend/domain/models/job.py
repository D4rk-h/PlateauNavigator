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
    CUSTOM = auto()  # todo: under consideration, not yet implemented

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
        if self.energy_history and len(self.energy_history) != self.iterations:
            raise ValueError(
                f"energy_history has {len(self.energy_history)} entries "
                f"but iterations is {self.iterations}"
            )
        if self.gradient_norm_history and len(self.gradient_norm_history) != self.iterations:
            raise ValueError(
                f"gradient_norm_history has {len(self.gradient_norm_history)} entries "
                f"but iterations is {self.iterations}"
            )

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

    def is_ready_to_run(self) -> bool:
        return self.status == JobStatus.PENDING

    def duration_seconds(self) -> Optional[float]:
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None