import pytest
from datetime import datetime
from unittest.mock import MagicMock

from backend.domain.models.job import (
    Job,
    JobConfig,
    JobResult,
    JobStatus,
    OptimizerType,
)
from backend.domain.models.hamiltonian import Paradigm


def make_hamiltonian(paradigm=Paradigm.DV, n_sites=4):
    h = MagicMock()
    h.paradigm = paradigm
    h.n_sites = n_sites
    return h


def make_circuit(paradigm=Paradigm.DV, n_sites=4):
    c = MagicMock()
    c.paradigm = paradigm
    c.n_sites = n_sites
    return c


def make_config(
    optimizer=OptimizerType.ADAM,
    max_iterations=100,
    learning_rate=0.01,
):
    return JobConfig(
        optimizer=optimizer,
        max_iterations=max_iterations,
        learning_rate=learning_rate,
    )


def make_job(**kwargs) -> Job:
    defaults = dict(
        name="test-job",
        hamiltonian=make_hamiltonian(),
        circuit=make_circuit(),
        config=make_config(),
    )
    defaults.update(kwargs)
    return Job(**defaults)


def make_result(**kwargs) -> JobResult:
    defaults = dict(final_energy=-1.5)
    defaults.update(kwargs)
    return JobResult(**defaults)

class TestJobConfig:

    def test_default_creation(self):
        c = make_config()
        assert c.optimizer == OptimizerType.ADAM
        assert c.max_iterations == 100

    def test_zero_max_iterations_fails(self):
        with pytest.raises(ValueError, match="max_iterations"):
            JobConfig(optimizer=OptimizerType.ADAM, max_iterations=0)

    def test_negative_max_iterations_fails(self):
        with pytest.raises(ValueError, match="max_iterations"):
            JobConfig(optimizer=OptimizerType.ADAM, max_iterations=-1)

    def test_zero_shots_fails(self):
        with pytest.raises(ValueError, match="shots"):
            JobConfig(optimizer=OptimizerType.ADAM, max_iterations=100, shots=0)

    def test_negative_learning_rate_fails(self):
        with pytest.raises(ValueError, match="learning_rate"):
            JobConfig(
                optimizer=OptimizerType.ADAM,
                max_iterations=100,
                learning_rate=-0.01,
            )

    def test_negative_convergence_tolerance_fails(self):
        with pytest.raises(ValueError, match="convergence_tolerance"):
            JobConfig(
                optimizer=OptimizerType.ADAM,
                max_iterations=100,
                convergence_tolerance=-1e-6,
            )

    def test_negative_gradient_samples_fails(self):
        with pytest.raises(ValueError, match="gradient_samples"):
            JobConfig(
                optimizer=OptimizerType.ADAM,
                max_iterations=100,
                gradient_samples=0,
            )

    def test_is_gradient_free_true_for_cobyla(self):
        c = JobConfig(optimizer=OptimizerType.COBYLA, max_iterations=100)
        assert c.is_gradient_free() is True

    def test_is_gradient_free_true_for_spsa(self):
        c = JobConfig(optimizer=OptimizerType.SPSA, max_iterations=100)
        assert c.is_gradient_free() is True

    def test_is_gradient_free_false_for_adam(self):
        c = make_config(optimizer=OptimizerType.ADAM)
        assert c.is_gradient_free() is False

    def test_is_gradient_free_false_for_gradient_descent(self):
        c = make_config(optimizer=OptimizerType.GRADIENT_DESCENT)
        assert c.is_gradient_free() is False

    def test_none_optional_fields_are_valid(self):
        c = JobConfig(optimizer=OptimizerType.ADAM, max_iterations=100)
        assert c.learning_rate is None
        assert c.shots is None
        assert c.random_seed is None
        assert c.convergence_tolerance is None
        assert c.gradient_samples is None

class TestJobResult:

    def test_default_creation(self):
        r = make_result()
        assert r.final_energy == -1.5
        assert r.iterations == 0
        assert r.converged is False
        assert r.has_error() is False

    def test_negative_iterations_fails(self):
        with pytest.raises(ValueError, match="iterations"):
            JobResult(final_energy=0.0, iterations=-1)

    def test_negative_execution_time_fails(self):
        with pytest.raises(ValueError, match="execution_time"):
            JobResult(final_energy=0.0, execution_time_seconds=-1.0)

    def test_negative_function_evaluations_fails(self):
        with pytest.raises(ValueError, match="function_evaluations"):
            JobResult(final_energy=0.0, function_evaluations=-1)

    def test_best_energy_from_history(self):
        r = make_result(
            final_energy=-1.0,
            energy_history=[-0.5, -0.8, -1.2, -1.0],
        )
        assert r.best_energy() == -1.2

    def test_best_energy_falls_back_to_final(self):
        r = make_result(final_energy=-1.5)
        assert r.best_energy() == -1.5

    def test_convergence_rate_with_history(self):
        r = make_result(
            final_energy=-1.0,
            energy_history=[0.0, -0.5, -1.0],
            iterations=3,
        )
        rate = r.convergence_rate()
        assert rate is not None
        assert rate > 0

    def test_convergence_rate_none_with_single_entry(self):
        r = make_result(energy_history=[-1.0], iterations=1)
        assert r.convergence_rate() is None

    def test_convergence_rate_none_with_no_history(self):
        r = make_result()
        assert r.convergence_rate() is None

    def test_has_error_true(self):
        r = make_result(error_message="backend timeout")
        assert r.has_error() is True

    def test_has_error_false(self):
        r = make_result()
        assert r.has_error() is False

    def test_summary_has_expected_keys(self):
        r = make_result(
            final_energy=-1.5,
            iterations=100,
            converged=True,
        )
        summary = r.summary()
        assert "final_energy" in summary
        assert "best_energy" in summary
        assert "iterations" in summary
        assert "function_evaluations" in summary
        assert "converged" in summary
        assert "execution_time_seconds" in summary
        assert "convergence_rate" in summary
        assert "has_error" in summary

    def test_energy_history_longer_than_iterations_is_valid(self):
        """Optimizers may log iteration 0 + each step — no strict equality."""
        r = JobResult(
            final_energy=-1.0,
            energy_history=[-0.5, -0.8, -1.0, -1.0],
            iterations=3,
        )
        assert len(r.energy_history) == 4
        assert r.iterations == 3

class TestJobCreation:

    def test_default_creation(self):
        job = make_job()
        assert job.status == JobStatus.PENDING
        assert job.result is None
        assert job.started_at is None
        assert job.completed_at is None

    def test_unique_ids(self):
        j1 = make_job()
        j2 = make_job()
        assert j1.id != j2.id

    def test_paradigm_mismatch_fails(self):
        with pytest.raises(ValueError, match="paradigm"):
            Job(
                name="bad",
                hamiltonian=make_hamiltonian(paradigm=Paradigm.DV),
                circuit=make_circuit(paradigm=Paradigm.CV),
                config=make_config(),
            )

    def test_n_sites_mismatch_fails(self):
        with pytest.raises(ValueError, match="n_sites"):
            Job(
                name="bad",
                hamiltonian=make_hamiltonian(n_sites=4),
                circuit=make_circuit(n_sites=2),
                config=make_config(),
            )

    def test_result_with_pending_status_fails(self):
        with pytest.raises(ValueError, match="PENDING"):
            Job(
                name="bad",
                hamiltonian=make_hamiltonian(),
                circuit=make_circuit(),
                config=make_config(),
                status=JobStatus.PENDING,
                result=make_result(),
            )

    def test_completed_at_without_started_at_fails(self):
        with pytest.raises(ValueError, match="started_at"):
            Job(
                name="bad",
                hamiltonian=make_hamiltonian(),
                circuit=make_circuit(),
                config=make_config(),
                completed_at=datetime.now(),
            )

class TestJobStateTransitions:

    def test_mark_running_from_pending(self):
        job = make_job()
        job.mark_running()
        assert job.status == JobStatus.RUNNING
        assert job.started_at is not None

    def test_mark_running_from_non_pending_fails(self):
        job = make_job()
        job.mark_running()
        with pytest.raises(ValueError, match="PENDING"):
            job.mark_running()

    def test_mark_completed_from_running(self):
        job = make_job()
        job.mark_running()
        job.mark_completed(make_result())
        assert job.status == JobStatus.COMPLETED
        assert job.result is not None
        assert job.completed_at is not None

    def test_mark_completed_from_pending_fails(self):
        job = make_job()
        with pytest.raises(ValueError, match="RUNNING"):
            job.mark_completed(make_result())

    def test_mark_failed_from_pending(self):
        job = make_job()
        job.mark_failed("connection error")
        assert job.status == JobStatus.FAILED
        assert job.result is not None
        assert job.result.has_error() is True
        assert job.result.error_message == "connection error"

    def test_mark_failed_from_running(self):
        job = make_job()
        job.mark_running()
        job.mark_failed("timeout")
        assert job.status == JobStatus.FAILED

    def test_mark_failed_from_completed_fails(self):
        job = make_job()
        job.mark_running()
        job.mark_completed(make_result())
        with pytest.raises(ValueError):
            job.mark_failed("late error")

    def test_mark_cancelled_from_pending(self):
        job = make_job()
        job.mark_cancelled()
        assert job.status == JobStatus.CANCELLED
        assert job.completed_at is not None

    def test_mark_cancelled_from_running(self):
        job = make_job()
        job.mark_running()
        job.mark_cancelled()
        assert job.status == JobStatus.CANCELLED

    def test_mark_cancelled_from_completed_fails(self):
        job = make_job()
        job.mark_running()
        job.mark_completed(make_result())
        with pytest.raises(ValueError):
            job.mark_cancelled()

class TestJobQueries:

    def test_is_ready_to_run_true_when_pending(self):
        job = make_job()
        assert job.is_ready_to_run() is True

    def test_is_ready_to_run_false_when_running(self):
        job = make_job()
        job.mark_running()
        assert job.is_ready_to_run() is False

    def test_is_terminal_false_when_pending(self):
        job = make_job()
        assert job.is_terminal() is False

    def test_is_terminal_true_when_completed(self):
        job = make_job()
        job.mark_running()
        job.mark_completed(make_result())
        assert job.is_terminal() is True

    def test_is_terminal_true_when_failed(self):
        job = make_job()
        job.mark_failed("error")
        assert job.is_terminal() is True

    def test_is_terminal_true_when_cancelled(self):
        job = make_job()
        job.mark_cancelled()
        assert job.is_terminal() is True

    def test_duration_seconds_none_when_not_started(self):
        job = make_job()
        assert job.duration_seconds() is None

    def test_duration_seconds_positive_when_completed(self):
        job = make_job()
        job.mark_running()
        job.mark_completed(make_result())
        duration = job.duration_seconds()
        assert duration is not None
        assert duration >= 0.0

    def test_summary_has_expected_keys(self):
        job = make_job()
        summary = job.summary()
        assert "id" in summary
        assert "name" in summary
        assert "status" in summary
        assert "created_at" in summary
        assert "started_at" in summary
        assert "completed_at" in summary
        assert "duration_seconds" in summary
        assert "result" in summary

    def test_summary_result_none_when_no_result(self):
        job = make_job()
        assert job.summary()["result"] is None

    def test_summary_result_present_after_completion(self):
        job = make_job()
        job.mark_running()
        job.mark_completed(make_result(final_energy=-2.0))
        assert job.summary()["result"] is not None