from abc import ABC, abstractmethod
from backend.domain.models.circuit import Circuit
from backend.domain.models.job import JobResult


class CircuitRunner(ABC):

    @abstractmethod
    async def run(self, circuit: Circuit) -> JobResult:
        pass

    @abstractmethod
    async def is_available(self) -> bool:
        pass

    @property
    @abstractmethod
    def backend_name(self) -> str:
        pass

    @property
    @abstractmethod
    def is_local(self) -> bool:
        pass

class DVCircuitRunner(CircuitRunner, ABC):
    pass

class CVCircuitRunner(CircuitRunner, ABC):
    pass
