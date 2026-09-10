import asyncio
from dataclasses import dataclass

from backend.domain.models.ansatz import Ansatz
from backend.domain.models.circuit import Circuit
from backend.domain.models.experiment import ComparisonExperiment, SingleExperiment
from backend.domain.models.job import Job, JobConfig
from backend.domain.models.physical_model import PhysicalModel
from backend.domain.ports.circuit_runner import CVCircuitRunner, DVCircuitRunner
from backend.domain.ports.experiment_repository import ExperimentRepository
from backend.domain.services.circuit_comparator import CircuitComparatorService
from backend.domain.services.hamiltonian_builder import HamiltonianBuilderService


@dataclass
class CompareCVDVRequest:
    name: str
    model: PhysicalModel
    dv_ansatz: Ansatz
    cv_ansatz: Ansatz
    dv_circuit: Circuit
    cv_circuit: Circuit
    dv_config: JobConfig
    cv_config: JobConfig
    dv_backend: str
    cv_backend: str
    check_expressibility: bool = True
    expressibility_threshold: float = 0.05


@dataclass
class CompareCVDVResponse:
    experiment: ComparisonExperiment
    expressibility_warning: str | None = None


class CompareCVDVUseCase:

    def __init__(
        self,
        dv_runner: DVCircuitRunner,
        cv_runner: CVCircuitRunner,
        hamiltonian_builder: HamiltonianBuilderService,
        circuit_comparator: CircuitComparatorService,
        experiment_repo: ExperimentRepository,
    ):
        self._dv_runner = dv_runner
        self._cv_runner = cv_runner
        self._builder = hamiltonian_builder
        self._comparator = circuit_comparator
        self._repo = experiment_repo

    async def execute(
        self,
        request: CompareCVDVRequest,
    ) -> CompareCVDVResponse:

        hamiltonian = self._builder.build(request.model)
        expressibility_warning = None

        if request.check_expressibility:
            comparison = self._comparator.compare(
                request.dv_ansatz,
                request.cv_ansatz,
            )
            if not comparison["matched"]:
                expressibility_warning = comparison["warning"]

        dv_available, cv_available = await asyncio.gather(
            self._dv_runner.is_available(),
            self._cv_runner.is_available(),
        )
        if not dv_available:
            raise RuntimeError(f"DV backend '{self._dv_runner.backend_name}' aint available.")
        if not cv_available:
            raise RuntimeError(f"CV backend '{self._cv_runner.backend_name}' aint available.")

        dv_job = Job(
            name=f"{request.name}-dv",
            hamiltonian=hamiltonian,
            circuit=request.dv_circuit,
            config=request.dv_config,
        )
        cv_job = Job(
            name=f"{request.name}-cv",
            hamiltonian=hamiltonian,
            circuit=request.cv_circuit,
            config=request.cv_config,
        )
        dv_job.mark_running()
        cv_job.mark_running()
        dv_result, cv_result = await asyncio.gather(
            self._run_safe(self._dv_runner, request.dv_circuit),
            self._run_safe(self._cv_runner, request.cv_circuit),
        )

        if isinstance(dv_result, Exception):
            dv_job.mark_failed(str(dv_result))
        else:
            dv_job.mark_completed(dv_result)

        if isinstance(cv_result, Exception):
            cv_job.mark_failed(str(cv_result))
        else:
            cv_job.mark_completed(cv_result)

        dv_experiment = SingleExperiment(
            name=f"{request.name}-dv",
            model=request.model,
            ansatz=request.dv_ansatz,
            job=dv_job,
        )
        cv_experiment = SingleExperiment(
            name=f"{request.name}-cv",
            model=request.model,
            ansatz=request.cv_ansatz,
            job=cv_job,
        )
        comparison_experiment = ComparisonExperiment(
            name=request.name,
            model=request.model,
            dv_experiment=dv_experiment,
            cv_experiment=cv_experiment,
        )

        if comparison_experiment.is_complete():
            comparison_experiment.build_comparison_result()

        await self._repo.save_comparison(comparison_experiment)

        return CompareCVDVResponse(
            experiment=comparison_experiment,
            expressibility_warning=expressibility_warning,
        )

    async def _run_safe(self, runner, circuit) -> object:
        try:
            return await runner.run(circuit)
        except Exception as e:
            return e