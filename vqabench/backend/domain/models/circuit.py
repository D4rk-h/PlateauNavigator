from dataclasses import dataclass, field
from typing import List, Optional, ClassVar
from enum import Enum, auto
import uuid

from backend.domain.models.parameter import Parameter
from backend.domain.models.hamiltonian import Paradigm


class CircuitSourceType(Enum):
    QASM3 = auto()
    QISKIT = auto()
    PYTHON_MRMUSTARD = auto()
    PENNYLANE = auto()

@dataclass
class Circuit:
    name: str
    paradigm: Paradigm
    n_sites: int
    source: str
    source_type: CircuitSourceType
    parameters: List[Parameter] = field(default_factory=list)
    description: Optional[str] = None
    id: str = field(default_factory=lambda: str(uuid.uuid4()))

    _VALID_SOURCE_TYPES: ClassVar[dict] = {
        Paradigm.DV: {CircuitSourceType.QASM3, CircuitSourceType.QISKIT},
        Paradigm.CV: {CircuitSourceType.PYTHON_MRMUSTARD, CircuitSourceType.PENNYLANE},
    }

    def __post_init__(self):
        if self.n_sites <= 0:
            raise ValueError("n_sites must be a positive integer.")
        if not self.source or not self.source.strip():
            raise ValueError("source must be a non-empty string.")
        if not isinstance(self.paradigm, Paradigm):
            raise ValueError("paradigm must be an instance of Paradigm Enum.")

        valid_types = self._VALID_SOURCE_TYPES[self.paradigm]
        if self.source_type not in valid_types:
            raise ValueError(
                f"{self.paradigm.name} circuits must use one of "
                f"{[t.name for t in valid_types]}"
            )

    def __len__(self) -> int:
        return len(self.parameters)

    def is_parameterized(self) -> bool:
        return len(self.parameters) > 0

    def trainable_parameters(self) -> List[Parameter]:
        return [p for p in self.parameters if p.trainable]

    def n_trainable(self) -> int:
        return len(self.trainable_parameters())
        