import pytest
from datetime import datetime, timedelta

from backend.domain.models.job import (
    Job, JobConfig, JobResult, JobStatus, OptimizerType,
)
from backend.domain.models.hamiltonian import Hamiltonian, PauliTerm, Paradigm
from backend.domain.models.circuit import Circuit, CircuitSourceType
from backend.domain.models.parameter import Parameter


def _dv_hamiltonian():
    return Hamiltonian(
        name="Ising",
        paradigm=Paradigm.DV,
        n_sites=2,
        terms=[PauliTerm("ZI", 1.0), PauliTerm("IZ", 1.0)],
    )


def _dv_circuit():
    return Circuit(
        name="HEA",
        paradigm=Paradigm.DV,
        n_sites=2,
        source="ry(theta) q[0];",
        source_type=CircuitSourceType.QASM3,
        parameters=[Parameter(name="theta", value=0.5)],
    )


def _basic_config():
    return JobConfig(
        optimizer=OptimizerType.SPSA,
        max_iterations=100,
        learning_rate=0.01,
        shots=1024,
    )


class TestJobConfig:

    def test_valid_config(self):
        c = JobConfig(optimizer=OptimizerType.ADAM, max_iterations=50)
        assert c.max_iterations == 50
        assert c.shots is None

    def test_zero_max_iterations_fails(self):
        with pytest.raises(ValueError, match="positive"):
            JobConfig(optimizer=OptimizerType.COBYLA, max_iterations=0)

    def test_negative_shots_fails(self):
        with pytest.raises(ValueError, match="positive"):
            JobConfig(optimizer=OptimizerType.SPSA, max_iterations=10, shots=-1)

    def test_negative_learning_rate_fails(self):
        with pytest.raises(ValueError, match="positive"):
            JobConfig(
                optimizer=OptimizerType.GRADIENT_DESCENT,
                max_iterations=10,
                learning_rate=-0.1,
            )

    def test_negative_convergence_tolerance_fails(self):
        with pytest.raises(ValueError, match="positive"):
            JobConfig(
                optimizer=OptimizerType.L_BFGS_B,
                max_iterations=10,
                convergence_tolerance=0,
            )


class TestJobResult:

    def test_empty_result(self):
        r = JobResult(final_energy=-1.5)
        assert r.iterations == 0
        assert r.converged is False

    def test_result_with_history(self):
        r = JobResult(
            final_energy=-1.5,
            energy_history=[0.0, -0.5, -1.0, -1.5],
            gradient_norm_history=[1.0, 0.5, 0.1, 0.01],
            iterations=4,
            converged=True,
        )
        assert len(r.energy_history) == 4

    def test_history_length_mismatch_fails(self):
        with pytest.raises(ValueError, match="energy_history"):
            JobResult(
                final_energy=0.0,
                energy_history=[0.0, -0.5],
                iterations=3,
            )

    def test_negative_iterations_fails(self):
        with pytest.raises(ValueError, match="non-negative"):
            JobResult(final_energy=0.0, iterations=-1)

    def test_negative_execution_time_fails(self):
        with pytest.raises(ValueError, match="non-negative"):
            JobResult(final_energy=0.0, execution_time_seconds=-1.0)


class TestJobCreation:

    def test_create_basic_job(self):
        job = Job(
            name="Test-VQE",
            hamiltonian=_dv_hamiltonian(),
            circuit=_dv_circuit(),
            config=_basic_config(),
        )
        assert job.name == "Test-VQE"
        assert job.status == JobStatus.PENDING
        assert job.is_ready_to_run() is True
        assert job.result is None

    def test_paradigm_mismatch_fails(self):
        cv_circuit = Circuit(
            name="Gaussian",
            paradigm=Paradigm.CV,
            n_sites=2,
            source="def ansatz(): pass",
            source_type=CircuitSourceType.PYTHON_MRMUSTARD,
        )
        with pytest.raises(ValueError, match="paradigm"):
            Job(
                name="bad",
                hamiltonian=_dv_hamiltonian(),
                circuit=cv_circuit,
                config=_basic_config(),
            )

    def test_n_sites_mismatch_fails(self):
        circuit_3q = Circuit(
            name="HEA-3",
            paradigm=Paradigm.DV,
            n_sites=3,
            source="ry(theta) q[0];",
            source_type=CircuitSourceType.QASM3,
        )
        with pytest.raises(ValueError, match="n_sites"):
            Job(
                name="bad",
                hamiltonian=_dv_hamiltonian(),
                circuit=circuit_3q,
                config=_basic_config(),
            )

    def test_result_with_pending_status_fails(self):
        with pytest.raises(ValueError, match="PENDING"):
            Job(
                name="bad",
                hamiltonian=_dv_hamiltonian(),
                circuit=_dv_circuit(),
                config=_basic_config(),
                status=JobStatus.PENDING,
                result=JobResult(final_energy=0.0),
            )

    def test_completed_without_started_fails(self):
        with pytest.raises(ValueError, match="started_at"):
            Job(
                name="bad",
                hamiltonian=_dv_hamiltonian(),
                circuit=_dv_circuit(),
                config=_basic_config(),
                completed_at=datetime.now(),
            )


class TestJobUtils:

    def test_duration_calculation(self):
        now = datetime.now()
        job = Job(
            name="timed",
            hamiltonian=_dv_hamiltonian(),
            circuit=_dv_circuit(),
            config=_basic_config(),
            started_at=now,
            completed_at=now + timedelta(seconds=5.5),
        )
        assert job.duration_seconds() == pytest.approx(5.5, abs=0.01)

    def test_duration_without_dates(self):
        job = Job(
            name="untimed",
            hamiltonian=_dv_hamiltonian(),
            circuit=_dv_circuit(),
            config=_basic_config(),
        )
        assert job.duration_seconds() is None