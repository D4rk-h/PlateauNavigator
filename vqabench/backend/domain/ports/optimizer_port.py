from abc import ABC, abstractmethod

from vqabench.backend.domain.models.circuit import Circuit
from vqabench.backend.domain.models.parameter import Parameter
from vqabench.backend.domain.models.job import JobConfig, JobResult

class OptimizerPort(ABC):
    @abstractmethod
    def optimize(
        self, 
        cost_fn: callable,
        initial_parameters: list[Parameter],
        config: JobConfig,
    ) -> JobResult:
        pass

    @property
    @abstractmethod
    def optimizer_type(self) -> str:
        pass

    @property
    @abstractmethod
    def requires_gradients(self) -> bool:
        pass