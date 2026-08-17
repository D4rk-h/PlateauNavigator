from abc import ABC, abstractmethod
from typing import Optional
from backend.domain.models.experiment import (
    SingleExperiment,
    ComparisonExperiment,
)


class ExperimentRepository(ABC):
    @abstractmethod
    async def save_single(self, experiment: SingleExperiment) -> None:
        pass

    @abstractmethod
    async def save_comparison(self, experiment: ComparisonExperiment) -> None:
        pass

    @abstractmethod
    async def get_single(self, experiment_id: str) -> Optional[SingleExperiment]:
        pass

    @abstractmethod
    async def get_comparison(self, experiment_id: str) -> Optional[ComparisonExperiment]:
        pass

    @abstractmethod
    async def list_singles(self, limit: int = 50, offset: int = 0) -> list[SingleExperiment]:
        pass

    @abstractmethod
    async def list_comparisons(self, limit: int = 50, offset: int = 0) -> list[ComparisonExperiment]:
        pass

    @abstractmethod
    async def delete(self, experiment_id: str) -> None:
        pass