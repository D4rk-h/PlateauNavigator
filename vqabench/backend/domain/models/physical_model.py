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

@dataclass
class FermionicModel(PhysicalModel, ABC):
    def preferred_paradigm(self) -> Paradigm:
        return Paradigm.DV

@dataclass
class IsingModel(FermionicModel):
    j: float = 1.0
    h: float = 0.5
    boundary: BoundaryCondition = BoundaryCondition.OPEN
    geometry: LatticeGeometry = LatticeGeometry.CHAIN

    def _validate(self):
        if self.geometry == LatticeGeometry.SQUARE:
            side = int(self.n_sites ** 0.5)
            if side * side != self.n_sites:
                raise ValueError(f"n_sites={self.n_sites} is not a perfect square for SQUARE geometry")

    def n_interactions(self) -> int:
        if self.geometry == LatticeGeometry.CHAIN:
            n = self.n_sites
            return n if self.boundary == BoundaryCondition.PERIODIC else n - 1
        if self.geometry == LatticeGeometry.SQUARE:
            side = int(self.n_sites ** 0.5)
            bonds = 2 * side * (side - 1)
            if self.boundary == BoundaryCondition.PERIODIC:
                bonds += side
            return bonds
        return self.n_sites * (self.n_sites - 1) // 2 

    def is_critical(self) -> bool:
        return abs(self.h) == abs(self.j)

    def summary(self) -> dict:
        return {
            **super().summary(),
            "j": self.j,
            "h": self.h,
            "boundary": self.boundary.name,
            "geometry": self.geometry.name,
            "critical": self.is_critical(),
        }

@dataclass
class HeisenbergModel(FermionicModel):
    Jx: float = 1.0
    Jy: float = 1.0
    Jz: float = 1.0
    boundary: BoundaryCondition = BoundaryCondition.OPEN

    def _validate(self) -> None:
        pass

    def n_interactions(self) -> int:
        n = self.n_sites
        return n if self.boundary == BoundaryCondition.PERIODIC else n - 1

    def model_subtype(self) -> str:
        if self.Jx == self.Jy == self.Jz:
            return "XXX"
        if self.Jx == self.Jy:
            return "XXZ"
        return "XYZ"

    def summary(self) -> dict:
        return {
            **super().summary(),
            "Jx": self.Jx,
            "Jy": self.Jy,
            "Jz": self.Jz,
            "boundary": self.boundary.name,
            "subtype": self.model_subtype(),            
        }

@dataclass
class FermiHubbardModel(FermionicModel):
    t: float = 1.0
    u: float = 2.0
    mu: float = 0.0
    boundary: BoundaryCondition = BoundaryCondition.OPEN

    def _validate(self) -> None:
        if self.n_sites < 2:
            raise ValueError("Fermi-Hubbard requires at least 2 sites")

    def n_qubits(self) -> int:
        return 2 * self.n_sites

    def n_interactions(self) -> int:
        hopping = self.n_sites if self.boundary == BoundaryCondition.PERIODIC else self.n_sites - 1
        return 2 * hopping + self.n_sites

    def is_half_filling(self) -> bool:
        return self.mu == self.u / 2

    def summary(self) -> dict:
        return {
            **super().summary(),
            "t": self.t,
            "u": self.u,
            "mu": self.mu,
            "boundary": self.boundary.name,
            "n_qubits": self.n_qubits(),
            "half_filling": self.is_half_filling(),
        }

_KNOWN_MOLECULES: dict[str, int] = {
    "H2": 4,
    "LiH": 12,
    "H2O": 14,
    "BeH2": 14,
    "N2": 20,
}

_KNOWN_ELECTRONS: dict[str, int] = {
    "H2": 2,
    "LiH": 4,
    "H2O": 10,
    "BeH2": 6,
    "N2": 14,
}

@dataclass
class MoleculeModel(PhysicalModel):
    molecule_name: str = "H2"
    basis_set: str = "sto-3g"
    charge: int = 0
    multiplicity: int = 1
    bond_length: Optional[float] = None
    n_sites: int = 0

    def __post_init__(self):
        if self.molecule_name in _KNOWN_MOLECULES:
            self.n_sites = _KNOWN_MOLECULES[self.molecule_name]
        if self.n_sites <= 0:
            raise ValueError(f"Unknown molecule '{self.molecule_name}'. Provide n_sites explicitly.")
        super().__post_init__()

    def _validate(self) -> None:
        if not (-2 <= self.charge <= 2):
            raise ValueError("charge must be between -2 and 2")
        if self.multiplicity < 1:
            raise ValueError("multiplicity must be at least 1")
        if self.bond_length is not None and self.bond_length <= 0:
            raise ValueError("bond_length must be positive")

    def preferred_paradigm(self) -> Paradigm:
        return Paradigm.DV

    def n_interactions(self) -> int:
        n = self.n_sites
        return n + n * (n - 1) // 2 + n * (n - 1) * (n - 2) * (n - 3) // 24

    def n_electrons(self) -> int:
        return _KNOWN_ELECTRONS.get(self.molecule_name, self.n_sites // 2)

    def is_known(self) -> bool:
        return self.molecule_name in _KNOWN_MOLECULES

    def summary(self) -> dict:
        return {
            **super().summary(),
            "molecule": self.molecule_name,
            "basis_set": self.basis_set,
            "charge": self.charge,
            "multiplicity": self.multiplicity,
            "bond_length": self.bond_length,
            "n_electrons": self.n_electrons(),
            "known": self.is_known(),
        }