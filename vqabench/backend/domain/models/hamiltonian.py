from dataclasses import dataclass, field
from typing import List, Union, ClassVar
from enum import Enum, auto
import uuid

from backend.domain.models.parameter import Parameter


class Paradigm(Enum):
    DV = auto()
    CV = auto()

@dataclass
class PauliTerm:
    pauli_string: str
    coefficient: float = 1.0

    def __post_init__(self):
        if not self.pauli_string:
            raise ValueError("pauli_string cannot be empty")
        valid = set("IXYZ")
        if not all(c in valid for c in self.pauli_string):
            invalid = set(self.pauli_string) - valid
            raise ValueError(f"pauli_string contains invalid characters: {invalid}.")
        
@dataclass
class CVTerm:
    operator: str
    modes: List[int]
    coefficient: float = 1.0
    power: int = 1

    def __post_init__(self):
        if not self.modes:
            raise ValueError("modes cannot be empty")
        if any(m < 0 for m in self.modes):
            raise ValueError("modes must be non-negative integers")
        if self.power < 1:
            raise ValueError("power must be a positive integer")

        valid_ops = {"n", "x", "p", "a", "a_dag", "adag"}
        if self.operator not in valid_ops:
            raise ValueError(f"operator must be one of {valid_ops}, got '{self.operator}'.")

@dataclass
class Hamiltonian:
    _PARADIGM_TYPES: ClassVar[dict] = {
        Paradigm.DV: PauliTerm,
        Paradigm.CV: CVTerm,
    }
    name: str
    paradigm: Paradigm
    n_sites: int
    terms: List[Union[PauliTerm, CVTerm]] = field(default_factory=list)
    parameters: List[Parameter] = field(default_factory=list)
    id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def __post_init__(self):
        if self.n_sites <= 0:
            raise ValueError("n_sites must be a positive integer")

        if not isinstance(self.paradigm, Paradigm):
            raise ValueError("paradigm must be an instance of Paradigm Enum")
        
        if not self.terms:
            raise ValueError("Hamiltonian must have at least one term")
        
        expected_type = self._PARADIGM_TYPES[self.paradigm]
        if not all(isinstance(t, expected_type) for t in self.terms):
            raise ValueError(f"All terms must be {expected_type.__name__} for {self.paradigm}")

        if self.paradigm == Paradigm.DV:
            for term in self.terms:
                if len(term.pauli_string) != self.n_sites:
                    raise ValueError(
                        f"Pauli string '{term.pauli_string}' has length {len(term.pauli_string)} but n_sites is {self.n_sites}")

        if self.paradigm == Paradigm.CV:
            for term in self.terms:
                max_mode = max(term.modes)
                if max_mode >= self.n_sites:
                    raise ValueError(f"mode index {max_mode} out of range for {self.n_sites} modes")

    def __len__(self) -> int:
        return len(self.terms)

    def is_parameterized(self) -> bool:
        return len(self.parameters) > 0

    def total_coefficient_norm(self) -> float:
        return sum(abs(term.coefficient) for term in self.terms)