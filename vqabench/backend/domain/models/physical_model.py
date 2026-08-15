from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum, auto
import uuid

from backend.domain.models.hamiltonian import Paradigm


class BoundaryCondition(Enum):
    PERIODIC = auto()
    OPEN = auto()

class LatticeGeometry(Enum):
    CHAIN = auto()
    SQUARE = auto()
    TRIANGULAR = auto()

@dataclass
class PhysicalModel(ABC):
    """
    Stores only physical parameters.
    Hamiltonian construction is delegated to HamiltonianBuilderService.
    """
    name: str
    n_sites: int
    id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def __post_init__(self):
        if self.n_sites <= 0:
            raise ValueError("n_sites must be positive")
        self._validate()

    @abstractmethod
    def _validate(self):
        pass

    @abstractmethod
    def preferred_paradigm(self) -> Paradigm:
        pass

    @abstractmethod
    def n_interactions(self) -> int:
        pass

    def summary(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "model": self.__class__.__name__,
            "n_sites": self.n_sites,
            "preferred_paradigm": self.preferred_paradigm().name,
            "n_interactions": self.n_interactions(),
        }

@dataclass
class BosonicModel(PhysicalModel, ABC):
    max_occupation: int = 2

    def preferred_paradigm(self) -> Paradigm:
        return Paradigm.CV

@dataclass
class BoseHubbardModel(BosonicModel):
    t: float = 1.0
    u: float = 1.0
    mu: float = 0.0
    boundary: BoundaryCondition = BoundaryCondition.OPEN

    def _validate(self):
        if self.u < 0:
            raise ValueError("Interaction strength u must be non-negative")
        if self.max_occupation < 1:
            raise ValueError("max_occupation must be at least 1")

    def n_interactions(self) -> int:
        n = self.n_sites
        return n if self.boundary == BoundaryCondition.PERIODIC else n - 1

    def is_superfluid(self) -> bool:
        return self.t > self.u / 2

    def is_mott_insulator(self) -> bool:
        return self.u > 10 * self.t

    def summary(self) -> dict:
        return {
            **super().summary(),
            "t": self.t,
            "u": self.u,
            "mu": self.mu,
            "boundary": self.boundary.name,
            "max_occupation": self.max_occupation,
        }

@dataclass
class KerrOscillatorModel(BosonicModel):
    omega: float = 1.0
    chi: float = 0.1

    def __post_init__(self):
        self.n_sites = 1
        super().__post_init__()

    def _validate(self):
        if self.omega <= 0:
            raise ValueError("omega must be positive")

    def n_interactions(self) -> int:
        return 1

    def is_non_gaussian(self) -> bool:
        return self.chi != 0.0

    def summary(self) -> dict:
        return {
            **super().summary(),
            "omega": self.omega,
            "chi": self.chi,
            "non_gaussian": self.is_non_gaussian(),
        }