from dataclasses import dataclass

from backend.domain.models.ansatz import Ansatz
from backend.domain.models.circuit import Circuit
from backend.domain.models.experiment import SingleExperiment
from backend.domain.models.job import Job, JobConfig
from backend.domain.models.physical_model import PhysicalModel
from backend.domain.ports.circuit_runner import CVCircuitRunner
from backend.domain.ports.experiment_repository import ExperimentRepository
from backend.domain.services.hamiltonian_builder import HamiltonianBuilderService


@dataclass
class RunCVExperimentRequest:
    name: str
    model: PhysicalModel
    ansatz: Ansatz
    circuit: Circuit
    config: JobConfig
    backend: str

@dataclass
class RunCVExperimentResponse:
    experiment: SingleExperiment

class RunCVExperimentUseCase:

    def __init__(
        self,
        runner: CVCircuitRunner,
        hamiltonian_builder: HamiltonianBuilderService,
        experiment_repo: ExperimentRepository,
    ):
        self._runner = runner
        self._builder = hamiltonian_builder
        self._repo = experiment_repo

    async def execute(self, request: RunCVExperimentRequest,) -> RunCVExperimentResponse:

        hamiltonian = self._builder.build(request.model)

        if not await self._runner.is_available():
            raise RuntimeError(f"Backend '{self._runner.backend_name}' aint available.")

        job = Job(
            name=request.name,
            hamiltonian=hamiltonian,
            circuit=request.circuit,
            config=request.config,
        )

        job.mark_running()
        try:
            result = await self._runner.run(request.circuit)
            job.mark_completed(result)
        except Exception as e:
            job.mark_failed(str(e))

        experiment = SingleExperiment(
            name=request.name,
            model=request.model,
            ansatz=request.ansatz,
            job=job,
        )
        await self._repo.save_single(experiment)

        return RunCVExperimentResponse(experiment=experiment)