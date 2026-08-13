from dataclasses import dataclass, field
from typing import List, Optional, ClassVar
from enum import Enum, auto
import uuid


from backend.domain.models.parameter import Parameter
from backend.domain.models.hamiltonian import Paradigm


class AnsatzType(Enum):
    HARDWARE_EFFICIENT = auto()
    UNITARY_COUPLED_CLUSTER = auto()
    ALTERNATING_LAYERED = auto()
    TENSOR_NETWORK = auto()
    GAUSSIAN = auto()
    NON_GAUSSIAN = auto()
    CUSTOM = auto()

@dataclass
class Ansatz:
    name: str
    paradigm: Paradigm
    n_sites: int
    ansatz_type: AnsatzType
    n_layers: int = 1
    entanglement_pattern: Optional[str] = None
    native_gates: List[str] = field(default_factory=list)
    parameters: List[Parameter] = field(default_factory=list)
    circuit_reference: Optional[str] = None

    non_gaussian_ops_per_layer: int = 0
    fock_cutoff: Optional[int] = None

    id: str = field(default_factory=lambda: str(uuid.uuid4()))

    _DV_TYPES: ClassVar[set] = {
        AnsatzType.HARDWARE_EFFICIENT,
        AnsatzType.UNITARY_COUPLED_CLUSTER,
        AnsatzType.ALTERNATING_LAYERED,
        AnsatzType.TENSOR_NETWORK,
        AnsatzType.CUSTOM,
    }

    _CV_TYPES: ClassVar[set] = {
        AnsatzType.GAUSSIAN, 
        AnsatzType.NON_GAUSSIAN,
        AnsatzType.CUSTOM
    }

    def __post_init__(self):
        if self.n_sites <= 0:
            raise ValueError("n_sites must be positive")
        if self.n_layers <= 0:
            raise ValueError("n_layers must be positive")
        if not isinstance(self.paradigm, Paradigm):
            raise ValueError("paradigm must be Paradigm enum")
        if not isinstance(self.ansatz_type, AnsatzType):
            raise ValueError("ansatz_type must be AnsatzType enum")
        if self.paradigm == Paradigm.DV and self.ansatz_type not in self._DV_TYPES:
            raise ValueError(
                f"AnsatzType {self.ansatz_type.name} not valid for DV paradigms"
            )
        if self.paradigm == Paradigm.CV and self.ansatz_type not in self._CV_TYPES:
            raise ValueError(
                f"AnsatzType {self.ansatz_type.name} not valid for CV paradigms"
            )

    def __len__(self):
        return len(self.parameters)

    def n_trainable(self) -> int:
        return sum(1 for p in self.parameters if p.trainable)

    def estimated_depth(self) -> int:
        if self.paradigm == Paradigm.CV:
            raise NotImplementedError(
                "Depth is not well-defined for CV ansatze. "
                "Use expressibility metrics instead."
            )
        gates_per_layer = 2 * self.n_sites
        entanglement_gates = {
            "linear": self.n_sites - 1,
            "circular": self.n_sites,
            "full": self.n_sites * (self.n_sites - 1) // 2,
        }

        if self.entanglement_pattern:
            gates_per_layer += entanglement_gates.get(
                self.entanglement_pattern,
                self.n_sites - 1
            )

        return gates_per_layer * self.n_layers

    def cv_complexity(self) -> dict:
        if self.paradigm == Paradigm.DV:
            raise ValueError(
                "CV complexity is only defined for CV ansatze."
            )
        gaussian_ops = self.n_layers * self.n_sites
        non_gaussian_ops = self.n_layers * self.non_gaussian_ops_per_layer
        return {
            "gaussian_ops": gaussian_ops,
            "non_gaussian_ops": non_gaussian_ops,
            "has_universal": non_gaussian_ops > 0,
        }

    def has_non_gaussian(self) -> bool:
        return self.paradigm == Paradigm.CV and self.non_gaussian_ops_per_layer > 0

    def expressibility_score(self) -> float:
        """
        Placeholder for ansatz expressibility estimation.

        For DV: based on Haar random fidelity distrbution
        For CV: based on Wigner function fidelity in truncated Fock space, weighted by non-Gaussian operations
        """
        raise NotImplementedError("Expressibility computation requires the expressibility service.")