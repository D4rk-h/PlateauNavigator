import numpy as np
from unittest.mock import MagicMock

from backend.domain.models.ansatz import Ansatz, AnsatzType
from backend.domain.models.hamiltonian import Paradigm
from backend.domain.models.expressibility import (
    ExpressibilityResult,
    DVExpressibilityResult,
    CVExpressibilityResult,
    ExpressibilityMethod,
)
from backend.domain.services.expressibility import ExpressibilityService


def make_dv_ansatz(n_sites: int = 2, n_layers: int = 1) -> Ansatz:
    return Ansatz(
        name="test-dv",
        paradigm=Paradigm.DV,
        n_sites=n_sites,
        ansatz_type=AnsatzType.HARDWARE_EFFICIENT,
        n_layers=n_layers,
    )

def make_cv_ansatz(n_sites: int = 2, fock_cutoff: int = 5) -> Ansatz:
    return Ansatz(
        name="test-cv",
        paradigm=Paradigm.CV,
        n_sites=n_sites,
        ansatz_type=AnsatzType.GAUSSIAN,
        fock_cutoff=fock_cutoff,
    )

def make_non_gaussian_ansatz(n_sites: int = 2, ops_per_layer: int = 1) -> Ansatz:
    return Ansatz(
        name="test-non-gaussian",
        paradigm=Paradigm.CV,
        n_sites=n_sites,
        ansatz_type=AnsatzType.NON_GAUSSIAN,
        non_gaussian_ops_per_layer=ops_per_layer,
    )

def make_service(fidelities: list[float], n_bins: int = 75) -> ExpressibilityService:
    sampler = MagicMock()
    sampler.sample_fidelities.return_value = fidelities
    return ExpressibilityService(sampler=sampler, n_bins=n_bins)

def uniform_fidelities(n: int = 1000) -> list[float]:
    return list(np.random.uniform(0, 1, n))

def haar_fidelities_dv(n_qubits: int, n: int = 1000, seed: int = 42) -> list[float]:
    hilbert_dim = 2 ** n_qubits
    rng = np.random.default_rng(seed)
    return rng.beta(1, hilbert_dim - 1, n).tolist()

class TestReturnTypes:
    def test_dv_ansatz_returns_dv_result(self):
        service = make_service(uniform_fidelities())
        result = service.compute(make_dv_ansatz())
        assert isinstance(result, DVExpressibilityResult)

    def test_cv_ansatz_returns_cv_result(self):
        service = make_service(uniform_fidelities())
        result = service.compute(make_cv_ansatz())
        assert isinstance(result, CVExpressibilityResult)

    def test_dv_result_is_subclass_of_base(self):
        service = make_service(uniform_fidelities())
        result = service.compute(make_dv_ansatz())
        assert isinstance(result, ExpressibilityResult)

    def test_cv_result_is_subclass_of_base(self):
        service = make_service(uniform_fidelities())
        result = service.compute(make_cv_ansatz())
        assert isinstance(result, ExpressibilityResult)

class TestDVExpressibility:
    def test_method_is_kl_divergence(self):
        service = make_service(uniform_fidelities())
        result = service.compute(make_dv_ansatz())
        assert result.method == ExpressibilityMethod.KL_DIVERGENCE

    def test_score_is_non_negative(self):
        service = make_service(uniform_fidelities())
        result = service.compute(make_dv_ansatz())
        assert result.score >= 0.0

    def test_n_qubits_stored_correctly(self):
        service = make_service(uniform_fidelities())
        result = service.compute(make_dv_ansatz(n_sites=3))
        assert result.n_qubits == 3

    def test_hilbert_dim_is_2_pow_n_qubits(self):
        service = make_service(uniform_fidelities())
        result = service.compute(make_dv_ansatz(n_sites=3))
        assert result.hilbert_dim == 2 ** 3

    def test_ansatz_id_stored_correctly(self):
        ansatz = make_dv_ansatz()
        service = make_service(uniform_fidelities())
        result = service.compute(ansatz)
        assert result.ansatz_id == ansatz.id

    def test_n_samples_stored_correctly(self):
        service = make_service(uniform_fidelities(500))
        result = service.compute(make_dv_ansatz(), n_samples=500)
        assert result.n_samples == 500

    def test_haar_fidelities_give_low_kl(self):
        fidelities = haar_fidelities_dv(n_qubits=2, n=2000)
        service = make_service(fidelities)
        result = service.compute(make_dv_ansatz(n_sites=2), n_samples=2000)
        assert result.score < 0.5

    def test_uniform_fidelities_give_high_kl_for_large_hilbert(self):
        service = make_service(uniform_fidelities(1000))
        result = service.compute(make_dv_ansatz(n_sites=4))
        assert result.score > 0.1

    def test_fidelity_distribution_sums_to_one(self):
        service = make_service(uniform_fidelities())
        result = service.compute(make_dv_ansatz())
        assert abs(sum(result.fidelity_distribution) - 1.0) < 1e-6

    def test_haar_distribution_sums_to_one(self):
        service = make_service(uniform_fidelities())
        result = service.compute(make_dv_ansatz())
        assert abs(sum(result.haar_distribution) - 1.0) < 1e-6

    def test_is_highly_expressible_false_for_uniform_large_hilbert(self):
        service = make_service(uniform_fidelities(1000))
        result = service.compute(make_dv_ansatz(n_sites=4))
        assert result.is_highly_expressible(threshold=0.05) is False

class TestDVSummary:
    def test_summary_has_n_qubits(self):
        service = make_service(uniform_fidelities())
        result = service.compute(make_dv_ansatz(n_sites=2))
        assert "n_qubits" in result.summary()

    def test_summary_has_hilbert_dim(self):
        service = make_service(uniform_fidelities())
        result = service.compute(make_dv_ansatz(n_sites=2))
        assert "hilbert_dim" in result.summary()

    def test_summary_has_no_cv_fields(self):
        service = make_service(uniform_fidelities())
        result = service.compute(make_dv_ansatz())
        summary = result.summary()
        assert "n_modes" not in summary
        assert "fock_cutoff" not in summary
        assert "has_non_gaussian" not in summary

    def test_summary_highly_expressible_is_bool(self):
        service = make_service(uniform_fidelities())
        result = service.compute(make_dv_ansatz())
        assert isinstance(result.summary()["highly_expressive"], bool)

class TestCVExpressibility:

    def test_method_is_haar_fock_truncated(self):
        service = make_service(uniform_fidelities())
        result = service.compute(make_cv_ansatz())
        assert result.method == ExpressibilityMethod.HAAR_FOCK_TRUNCATED

    def test_score_is_non_negative(self):
        service = make_service(uniform_fidelities())
        result = service.compute(make_cv_ansatz())
        assert result.score >= 0.0

    def test_fock_cutoff_stored_correctly(self):
        service = make_service(uniform_fidelities())
        result = service.compute(make_cv_ansatz(fock_cutoff=8))
        assert result.fock_cutoff == 8

    def test_default_cutoff_is_10_when_not_set(self):
        ansatz = Ansatz(
            name="no-cutoff",
            paradigm=Paradigm.CV,
            n_sites=2,
            ansatz_type=AnsatzType.GAUSSIAN,
        )
        service = make_service(uniform_fidelities())
        result = service.compute(ansatz)
        assert result.fock_cutoff == 10

    def test_n_modes_stored_correctly(self):
        service = make_service(uniform_fidelities())
        result = service.compute(make_cv_ansatz(n_sites=3))
        assert result.n_modes == 3

    def test_has_non_gaussian_false_for_gaussian_ansatz(self):
        service = make_service(uniform_fidelities())
        result = service.compute(make_cv_ansatz())
        assert result.has_non_gaussian is False

    def test_has_non_gaussian_true_for_non_gaussian_ansatz(self):
        service = make_service(uniform_fidelities())
        result = service.compute(make_non_gaussian_ansatz())
        assert result.has_non_gaussian is True

    def test_fidelity_distribution_sums_to_one(self):
        service = make_service(uniform_fidelities())
        result = service.compute(make_cv_ansatz())
        assert abs(sum(result.fidelity_distribution) - 1.0) < 1e-6

    def test_haar_distribution_sums_to_one(self):
        service = make_service(uniform_fidelities())
        result = service.compute(make_cv_ansatz())
        assert abs(sum(result.haar_distribution) - 1.0) < 1e-6

    def test_ansatz_id_stored_correctly(self):
        ansatz = make_cv_ansatz()
        service = make_service(uniform_fidelities())
        result = service.compute(ansatz)
        assert result.ansatz_id == ansatz.id

class TestCVSummary:

    def test_summary_has_n_modes(self):
        service = make_service(uniform_fidelities())
        result = service.compute(make_cv_ansatz())
        assert "n_modes" in result.summary()

    def test_summary_has_fock_cutoff(self):
        service = make_service(uniform_fidelities())
        result = service.compute(make_cv_ansatz())
        assert "fock_cutoff" in result.summary()

    def test_summary_has_has_non_gaussian(self):
        service = make_service(uniform_fidelities())
        result = service.compute(make_cv_ansatz())
        assert "has_non_gaussian" in result.summary()

    def test_summary_has_is_universal(self):
        service = make_service(uniform_fidelities())
        result = service.compute(make_non_gaussian_ansatz())
        assert "is_universal" in result.summary()

    def test_summary_is_universal_true_for_non_gaussian(self):
        service = make_service(uniform_fidelities())
        result = service.compute(make_non_gaussian_ansatz())
        assert result.summary()["is_universal"] is True

    def test_summary_is_universal_false_for_gaussian(self):
        service = make_service(uniform_fidelities())
        result = service.compute(make_cv_ansatz())
        assert result.summary()["is_universal"] is False

    def test_summary_has_no_dv_fields(self):
        service = make_service(uniform_fidelities())
        result = service.compute(make_cv_ansatz())
        summary = result.summary()
        assert "n_qubits" not in summary
        assert "hilbert_dim" not in summary

class TestCompare:

    def test_compare_returns_expected_keys(self):
        service = make_service(uniform_fidelities())
        result = service.compare(make_dv_ansatz(), make_cv_ansatz())
        assert {"dv", "cv", "delta", "matched"} == set(result.keys())

    def test_delta_is_non_negative(self):
        service = make_service(uniform_fidelities())
        result = service.compare(make_dv_ansatz(), make_cv_ansatz())
        assert result["delta"] >= 0.0

    def test_matched_is_bool(self):
        service = make_service(uniform_fidelities())
        result = service.compare(make_dv_ansatz(), make_cv_ansatz())
        assert isinstance(result["matched"], bool)

    def test_dv_summary_in_compare_has_dv_fields(self):
        service = make_service(uniform_fidelities())
        result = service.compare(make_dv_ansatz(), make_cv_ansatz())
        assert "n_qubits" in result["dv"]

    def test_cv_summary_in_compare_has_cv_fields(self):
        service = make_service(uniform_fidelities())
        result = service.compare(make_dv_ansatz(), make_cv_ansatz())
        assert "n_modes" in result["cv"]

    def test_matched_true_when_scores_identical(self):
        """Same fidelities for both → same score → delta 0 → matched."""
        fidelities = uniform_fidelities(500)
        sampler = MagicMock()
        sampler.sample_fidelities.return_value = fidelities
        service = ExpressibilityService(sampler=sampler, n_bins=75)
        result = service.compare(
            make_dv_ansatz(n_sites=2),
            make_cv_ansatz(n_sites=2, fock_cutoff=2),
        )
        assert isinstance(result["matched"], bool)

class TestPureMath:

    def test_kl_divergence_identical_distributions_is_zero(self):
        service = make_service([])
        p = np.array([0.25, 0.25, 0.25, 0.25])
        assert service._kl_divergence(p, p) < 1e-10

    def test_kl_divergence_non_negative(self):
        service = make_service([])
        p = np.array([0.5, 0.3, 0.2])
        q = np.array([0.2, 0.3, 0.5])
        assert service._kl_divergence(p, q) >= 0.0

    def test_kl_divergence_not_symmetric(self):
        service = make_service([])
        p = np.array([0.5, 0.3, 0.2])
        q = np.array([0.1, 0.6, 0.5])
        assert service._kl_divergence(p, q) != service._kl_divergence(q, p)

    def test_build_distribution_sums_to_one(self):
        service = make_service([])
        dist = service._build_distribution(uniform_fidelities(1000))
        assert abs(dist.sum() - 1.0) < 1e-6

    def test_build_distribution_clips_below_zero(self):
        service = make_service([])
        dist = service._build_distribution([-0.05, 0.5, 0.3])
        assert dist.sum() > 0

    def test_build_distribution_clips_above_one(self):
        service = make_service([])
        dist = service._build_distribution([0.5, 1.05, 0.3])
        assert dist.sum() > 0

    def test_haar_dv_sums_to_one(self):
        service = make_service([])
        haar = service._haar_distribution_dv(hilbert_dim=4)
        assert abs(haar.sum() - 1.0) < 1e-6

    def test_haar_dv_larger_hilbert_concentrates_near_zero(self):
        service = make_service([])
        haar_small = service._haar_distribution_dv(hilbert_dim=4)
        haar_large = service._haar_distribution_dv(hilbert_dim=256)
        assert haar_large[0] > haar_small[0]

    def test_haar_cv_sums_to_one(self):
        service = make_service([])
        haar = service._haar_distribution_cv(hilbert_dim=25)
        assert abs(haar.sum() - 1.0) < 1e-6