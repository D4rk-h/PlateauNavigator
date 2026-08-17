import pytest
from unittest.mock import MagicMock
from datetime import datetime

from backend.domain.models.experiment import (
    SingleExperiment,
    ComparisonExperiment,
    ComparisonResult,
    ExperimentStatus,
)
from backend.domain.models.job import Job, JobStatus, JobResult, JobConfig, OptimizerType
from backend.domain.models.ansatz import Ansatz, AnsatzType
from backend.domain.models.hamiltonian import Paradigm
from backend.domain.models.expressibility import (
    DVExpressibilityResult,
    ExpressibilityMethod,
    CVExpressibilityResult,
)

def make_mock_model(n_sites: int = 4):
    m = MagicMock()
    m.n_sites = n_sites
    m.__class__.__name__ = "BoseHubbardModel"
    return m


def make_mock_circuit(paradigm: Paradigm, n_sites: int = 4):
    c = MagicMock()
    c.paradigm = paradigm
    c.n_sites = n_sites
    return c


def make_mock_hamiltonian(paradigm: Paradigm, n_sites: int = 4):
    h = MagicMock()
    h.paradigm = paradigm
    h.n_sites = n_sites
    return h


def make_config() -> JobConfig:
    return JobConfig(optimizer=OptimizerType.ADAM, max_iterations=100)


def make_job(paradigm: Paradigm = Paradigm.DV, n_sites: int = 4) -> Job:
    return Job(
        name="test-job",
        hamiltonian=make_mock_hamiltonian(paradigm, n_sites),
        circuit=make_mock_circuit(paradigm, n_sites),
        config=make_config(),
    )


def make_ansatz(paradigm: Paradigm, n_sites: int = 4) -> Ansatz:
    ansatz_type = (
        AnsatzType.HARDWARE_EFFICIENT
        if paradigm == Paradigm.DV
        else AnsatzType.GAUSSIAN
    )
    return Ansatz(
        name="test-ansatz",
        paradigm=paradigm,
        n_sites=n_sites,
        ansatz_type=ansatz_type,
    )


def make_result(energy: float = -1.5, iterations: int = 100) -> JobResult:
    return JobResult(
        final_energy=energy,
        energy_history=[-0.5, -1.0, energy],
        iterations=iterations,
        converged=True,
    )


def make_dv_expr(score: float = 0.05) -> DVExpressibilityResult:
    return DVExpressibilityResult(
        ansatz_id="test",
        method=ExpressibilityMethod.KL_DIVERGENCE,
        score=score,
        n_samples=1000,
        n_bins=75,
        fidelity_distribution=[0.01] * 75,
        haar_distribution=[0.01] * 75,
        n_qubits=4,
        hilbert_dim=16,
    )


def make_cv_expr(score: float = 0.06) -> CVExpressibilityResult:
    return CVExpressibilityResult(
        ansatz_id="test",
        method=ExpressibilityMethod.HAAR_FOCK_TRUNCATED,
        score=score,
        n_samples=1000,
        n_bins=75,
        fidelity_distribution=[0.01] * 75,
        haar_distribution=[0.01] * 75,
        n_modes=4,
        fock_cutoff=10,
        has_non_gaussian=False,
    )


def make_single(
    paradigm: Paradigm = Paradigm.DV,
    n_sites: int = 4,
    job: Job = None,
    expressibility=None,
) -> SingleExperiment:
    return SingleExperiment(
        name="test-single",
        model=make_mock_model(n_sites),
        ansatz=make_ansatz(paradigm, n_sites),
        job=job or make_job(paradigm, n_sites),
        expressibility=expressibility,
    )


def make_comparison(
    n_sites: int = 4,
    dv: SingleExperiment = None,
    cv: SingleExperiment = None,
) -> ComparisonExperiment:
    return ComparisonExperiment(
        name="test-comparison",
        model=make_mock_model(n_sites),
        dv_experiment=dv or make_single(Paradigm.DV, n_sites),
        cv_experiment=cv or make_single(Paradigm.CV, n_sites),
    )

class TestSingleExperimentCreation:

    def test_default_creation_dv(self):
        exp = make_single(Paradigm.DV)
        assert exp.paradigm == Paradigm.DV
        assert exp.status == ExperimentStatus.CREATED

    def test_default_creation_cv(self):
        exp = make_single(Paradigm.CV)
        assert exp.paradigm == Paradigm.CV

    def test_model_ansatz_n_sites_mismatch_fails(self):
        with pytest.raises(ValueError, match="n_sites"):
            SingleExperiment(
                name="bad",
                model=make_mock_model(n_sites=4),
                ansatz=make_ansatz(Paradigm.DV, n_sites=2),
                job=make_job(Paradigm.DV, n_sites=2),
            )

    def test_ansatz_job_paradigm_mismatch_fails(self):
        with pytest.raises(ValueError, match="paradigm"):
            SingleExperiment(
                name="bad",
                model=make_mock_model(n_sites=4),
                ansatz=make_ansatz(Paradigm.DV, n_sites=4),
                job=make_job(Paradigm.CV, n_sites=4),
            )

    def test_unique_ids(self):
        e1 = make_single()
        e2 = make_single()
        assert e1.id != e2.id

class TestSingleExperimentStatus:

    def test_status_created_when_pending(self):
        exp = make_single()
        assert exp.status == ExperimentStatus.CREATED

    def test_status_running_when_job_running(self):
        job = make_job()
        job.mark_running()
        exp = make_single(job=job)
        assert exp.status == ExperimentStatus.RUNNING

    def test_status_completed_when_job_completed(self):
        job = make_job()
        job.mark_running()
        job.mark_completed(make_result())
        exp = make_single(job=job)
        assert exp.status == ExperimentStatus.COMPLETED

    def test_status_failed_when_job_failed(self):
        job = make_job()
        job.mark_failed("error")
        exp = make_single(job=job)
        assert exp.status == ExperimentStatus.FAILED

    def test_status_failed_when_job_cancelled(self):
        job = make_job()
        job.mark_cancelled()
        exp = make_single(job=job)
        assert exp.status == ExperimentStatus.FAILED

class TestSingleExperimentQueries:

    def test_is_complete_false_when_pending(self):
        exp = make_single()
        assert exp.is_complete() is False

    def test_is_complete_true_when_completed(self):
        job = make_job()
        job.mark_running()
        job.mark_completed(make_result())
        exp = make_single(job=job)
        assert exp.is_complete() is True

    def test_final_energy_none_when_not_complete(self):
        exp = make_single()
        assert exp.final_energy() is None

    def test_final_energy_present_when_complete(self):
        job = make_job()
        job.mark_running()
        job.mark_completed(make_result(energy=-2.0))
        exp = make_single(job=job)
        assert exp.final_energy() == -2.0

    def test_best_energy_none_when_not_complete(self):
        exp = make_single()
        assert exp.best_energy() is None

    def test_best_energy_from_history(self):
        job = make_job()
        job.mark_running()
        job.mark_completed(JobResult(
            final_energy=-1.0,
            energy_history=[-0.5, -1.5, -1.0],
            iterations=3,
        ))
        exp = make_single(job=job)
        assert exp.best_energy() == -1.5

    def test_has_expressibility_false_by_default(self):
        exp = make_single()
        assert exp.has_expressibility() is False

    def test_has_expressibility_true_when_set(self):
        exp = make_single(expressibility=make_dv_expr())
        assert exp.has_expressibility() is True

    def test_expressibility_score_none_when_not_set(self):
        exp = make_single()
        assert exp.expressibility_score() is None

    def test_expressibility_score_present_when_set(self):
        exp = make_single(expressibility=make_dv_expr(score=0.03))
        assert exp.expressibility_score() == pytest.approx(0.03)

    def test_summary_has_expected_keys(self):
        exp = make_single()
        summary = exp.summary()
        assert "id" in summary
        assert "name" in summary
        assert "status" in summary
        assert "paradigm" in summary
        assert "model" in summary
        assert "n_sites" in summary
        assert "ansatz" in summary
        assert "final_energy" in summary
        assert "best_energy" in summary
        assert "expressibility_score" in summary
        assert "created_at" in summary

class TestComparisonResult:

    def make_result(self, dv_energy=-1.5, cv_energy=-1.3) -> ComparisonResult:
        return ComparisonResult(
            dv_final_energy=dv_energy,
            cv_final_energy=cv_energy,
            dv_best_energy=dv_energy - 0.1,
            cv_best_energy=cv_energy - 0.1,
            dv_iterations=100,
            cv_iterations=120,
            dv_expressibility_score=0.04,
            cv_expressibility_score=0.05,
            expressibility_delta=0.01,
        )

    def test_energy_delta(self):
        r = self.make_result(dv_energy=-1.5, cv_energy=-1.3)
        assert r.energy_delta() == pytest.approx(0.2)

    def test_best_energy_delta(self):
        r = self.make_result(dv_energy=-1.5, cv_energy=-1.3)
        assert r.best_energy_delta() == pytest.approx(0.2)

    def test_winner_by_energy_dv(self):
        r = self.make_result(dv_energy=-1.5, cv_energy=-1.3)
        assert r.winner_by_energy() == "DV"

    def test_winner_by_energy_cv(self):
        r = self.make_result(dv_energy=-1.3, cv_energy=-1.5)
        assert r.winner_by_energy() == "CV"

    def test_winner_by_energy_tie(self):
        r = self.make_result(dv_energy=-1.5, cv_energy=-1.5)
        assert r.winner_by_energy() == "TIE"

    def test_winner_by_best_energy_dv(self):
        r = self.make_result(dv_energy=-1.5, cv_energy=-1.3)
        assert r.winner_by_best_energy() == "DV"

    def test_expressibility_matched_true(self):
        r = self.make_result()
        assert r.expressibility_matched(threshold=0.05) is True

    def test_expressibility_matched_false_when_delta_large(self):
        r = ComparisonResult(
            dv_final_energy=-1.5,
            cv_final_energy=-1.3,
            dv_best_energy=-1.6,
            cv_best_energy=-1.4,
            dv_iterations=100,
            cv_iterations=100,
            dv_expressibility_score=0.01,
            cv_expressibility_score=0.5,
            expressibility_delta=0.49,
        )
        assert r.expressibility_matched(threshold=0.05) is False

    def test_expressibility_matched_false_when_delta_none(self):
        r = ComparisonResult(
            dv_final_energy=-1.5,
            cv_final_energy=-1.3,
            dv_best_energy=-1.6,
            cv_best_energy=-1.4,
            dv_iterations=100,
            cv_iterations=100,
            dv_expressibility_score=None,
            cv_expressibility_score=None,
            expressibility_delta=None,
        )
        assert r.expressibility_matched() is False

    def test_summary_has_expected_keys(self):
        r = self.make_result()
        summary = r.summary()
        assert "dv_final_energy" in summary
        assert "cv_final_energy" in summary
        assert "energy_delta" in summary
        assert "winner_by_energy" in summary
        assert "winner_by_best_energy" in summary
        assert "expressibility_matched" in summary
        assert "expressibility_delta" in summary

class TestComparisonExperimentCreation:

    def test_default_creation(self):
        comp = make_comparison()
        assert comp.status == ExperimentStatus.CREATED

    def test_dv_experiment_with_cv_paradigm_fails(self):
        with pytest.raises(ValueError, match="DV paradigm"):
            ComparisonExperiment(
                name="bad",
                model=make_mock_model(),
                dv_experiment=make_single(Paradigm.CV),
                cv_experiment=make_single(Paradigm.CV),
            )

    def test_cv_experiment_with_dv_paradigm_fails(self):
        with pytest.raises(ValueError, match="CV paradigm"):
            ComparisonExperiment(
                name="bad",
                model=make_mock_model(),
                dv_experiment=make_single(Paradigm.DV),
                cv_experiment=make_single(Paradigm.DV),
            )

    def test_n_sites_mismatch_fails(self):
        with pytest.raises(ValueError, match="n_sites"):
            ComparisonExperiment(
                name="bad",
                model=make_mock_model(),
                dv_experiment=make_single(Paradigm.DV, n_sites=4),
                cv_experiment=make_single(Paradigm.CV, n_sites=2),
            )

    def test_unique_ids(self):
        c1 = make_comparison()
        c2 = make_comparison()
        assert c1.id != c2.id

class TestComparisonExperimentStatus:

    def test_status_created_when_both_pending(self):
        comp = make_comparison()
        assert comp.status == ExperimentStatus.CREATED

    def test_status_running_when_one_running(self):
        dv_job = make_job(Paradigm.DV)
        dv_job.mark_running()
        comp = make_comparison(
            dv=make_single(Paradigm.DV, job=dv_job),
        )
        assert comp.status == ExperimentStatus.RUNNING

    def test_status_completed_when_both_complete(self):
        dv_job = make_job(Paradigm.DV)
        dv_job.mark_running()
        dv_job.mark_completed(make_result())

        cv_job = make_job(Paradigm.CV)
        cv_job.mark_running()
        cv_job.mark_completed(make_result())

        comp = make_comparison(
            dv=make_single(Paradigm.DV, job=dv_job),
            cv=make_single(Paradigm.CV, job=cv_job),
        )
        assert comp.status == ExperimentStatus.COMPLETED

    def test_status_failed_when_one_fails(self):
        dv_job = make_job(Paradigm.DV)
        dv_job.mark_failed("error")
        comp = make_comparison(
            dv=make_single(Paradigm.DV, job=dv_job),
        )
        assert comp.status == ExperimentStatus.FAILED

class TestBuildComparisonResult:

    def make_completed_comparison(
        self,
        dv_expr=None,
        cv_expr=None,
    ) -> ComparisonExperiment:
        dv_job = make_job(Paradigm.DV)
        dv_job.mark_running()
        dv_job.mark_completed(make_result(energy=-1.5))

        cv_job = make_job(Paradigm.CV)
        cv_job.mark_running()
        cv_job.mark_completed(make_result(energy=-1.3))

        return make_comparison(
            dv=make_single(Paradigm.DV, job=dv_job, expressibility=dv_expr),
            cv=make_single(Paradigm.CV, job=cv_job, expressibility=cv_expr),
        )

    def test_build_comparison_result_returns_comparison_result(self):
        comp = self.make_completed_comparison()
        result = comp.build_comparison_result()
        assert isinstance(result, ComparisonResult)

    def test_build_sets_comparison_result_on_experiment(self):
        comp = self.make_completed_comparison()
        comp.build_comparison_result()
        assert comp.has_comparison_result() is True

    def test_build_energies_correct(self):
        comp = self.make_completed_comparison()
        result = comp.build_comparison_result()
        assert result.dv_final_energy == pytest.approx(-1.5)
        assert result.cv_final_energy == pytest.approx(-1.3)

    def test_build_winner_is_dv(self):
        comp = self.make_completed_comparison()
        result = comp.build_comparison_result()
        assert result.winner_by_energy() == "DV"

    def test_build_with_expressibility(self):
        comp = self.make_completed_comparison(
            dv_expr=make_dv_expr(score=0.04),
            cv_expr=make_cv_expr(score=0.05),
        )
        result = comp.build_comparison_result()
        assert result.dv_expressibility_score == pytest.approx(0.04)
        assert result.cv_expressibility_score == pytest.approx(0.05)
        assert result.expressibility_delta == pytest.approx(0.01)

    def test_build_without_expressibility_gives_none_delta(self):
        comp = self.make_completed_comparison()
        result = comp.build_comparison_result()
        assert result.expressibility_delta is None

    def test_build_fails_when_dv_not_complete(self):
        comp = make_comparison()
        with pytest.raises(ValueError, match="DV"):
            comp.build_comparison_result()

    def test_build_fails_when_cv_not_complete(self):
        dv_job = make_job(Paradigm.DV)
        dv_job.mark_running()
        dv_job.mark_completed(make_result())
        comp = make_comparison(
            dv=make_single(Paradigm.DV, job=dv_job),
        )
        with pytest.raises(ValueError, match="CV"):
            comp.build_comparison_result()

    def test_summary_has_comparison_result_after_build(self):
        comp = self.make_completed_comparison()
        comp.build_comparison_result()
        summary = comp.summary()
        assert summary["comparison_result"] is not None

    def test_summary_comparison_result_none_before_build(self):
        comp = make_comparison()
        assert comp.summary()["comparison_result"] is None
