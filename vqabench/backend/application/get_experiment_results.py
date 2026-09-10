from dataclasses import dataclass
from typing import Optional, Union
from enum import Enum, auto

from backend.domain.models.experiment import ComparisonExperiment, SingleExperiment
from backend.domain.ports.experiment_repository import ExperimentRepository


class ExperimentType(Enum):
    SINGLE = auto()
    COMPARISON = auto()

@dataclass
class GetExperimentResultsRequest:
    experiment_id: str
    experiment_type: ExperimentType

@dataclass
class GetExperimentResultsResponse:
    experiment: Union[SingleExperiment, ComparisonExperiment]


class GetExperimentResultsUseCase:

    def __init__(self, experiment_repo: ExperimentRepository):
        self._repo = experiment_repo

    async def execute(self, request: GetExperimentResultsRequest,) -> GetExperimentResultsResponse:

        if request.experiment_type == ExperimentType.SINGLE:
            experiment = await self._repo.get_single(request.experiment_id)
        elif request.experiment_type == ExperimentType.COMPARISON:
            experiment = await self._repo.get_comparison(request.experiment_id)
        else:
            raise ValueError(
                f"Invalid experiment_type '{request.experiment_type}'. "
                f"Must be 'single' or 'comparison'."
            )

        if experiment is None:
            raise ValueError(f"Experiment '{request.experiment_id}' not found.")

        return GetExperimentResultsResponse(experiment=experiment)