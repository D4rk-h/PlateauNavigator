from abc import ABC, abstractmethod
from backend.domain.models.ansatz import Ansatz

class ExpressibilityPort(ABC):

    @abstractmethod
    def sample_fidelities(self, ansatz: Ansatz, n_samples: int) -> list[float]:
        pass