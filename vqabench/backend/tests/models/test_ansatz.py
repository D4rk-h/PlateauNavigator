import pytest
from backend.domain.models.ansatz import Ansatz, AnsatzType
from backend.domain.models.hamiltonian import Paradigm
from backend.domain.models.parameter import Parameter


class TestAnsatzCreation:

    def test_create_dv_hardware_efficient(self):
        a = Ansatz(
            name="HEA-4",
            paradigm=Paradigm.DV,
            n_sites=4,
            ansatz_type=AnsatzType.HARDWARE_EFFICIENT,
            n_layers=2,
            entanglement_pattern="linear",
            native_gates=["ry", "rz", "cx"],
        )
        assert a.name == "HEA-4"
        assert a.paradigm == Paradigm.DV
        assert a.n_sites == 4
        assert a.n_layers == 2
        assert a.ansatz_type == AnsatzType.HARDWARE_EFFICIENT
        assert a.entanglement_pattern == "linear"

    def test_create_cv_gaussian(self):
        a = Ansatz(
            name="Gaussian-2",
            paradigm=Paradigm.CV,
            n_sites=2,
            ansatz_type=AnsatzType.GAUSSIAN,
            n_layers=1,
        )
        assert a.paradigm == Paradigm.CV
        assert a.ansatz_type == AnsatzType.GAUSSIAN

    def test_create_custom_both_paradigms(self):
        a_dv = Ansatz(
            name="custom-dv",
            paradigm=Paradigm.DV,
            n_sites=2,
            ansatz_type=AnsatzType.CUSTOM,
        )
        a_cv = Ansatz(
            name="custom-cv",
            paradigm=Paradigm.CV,
            n_sites=2,
            ansatz_type=AnsatzType.CUSTOM,
        )
        assert a_dv.ansatz_type == AnsatzType.CUSTOM
        assert a_cv.ansatz_type == AnsatzType.CUSTOM

    def test_ansatz_with_parameters(self):
        a = Ansatz(
            name="Parametric",
            paradigm=Paradigm.DV,
            n_sites=2,
            ansatz_type=AnsatzType.CUSTOM,
            parameters=[
                Parameter(name="theta", value=0.5),
                Parameter(name="phi", value=0.0, trainable=False),
            ],
        )
        assert len(a) == 2
        assert a.n_trainable() == 1


class TestAnsatzValidations:

    def test_zero_n_sites_fails(self):
        with pytest.raises(ValueError, match="positive"):
            Ansatz(
                name="bad",
                paradigm=Paradigm.DV,
                n_sites=0,
                ansatz_type=AnsatzType.HARDWARE_EFFICIENT,
            )

    def test_zero_n_layers_fails(self):
        with pytest.raises(ValueError, match="positive"):
            Ansatz(
                name="bad",
                paradigm=Paradigm.DV,
                n_sites=2,
                ansatz_type=AnsatzType.HARDWARE_EFFICIENT,
                n_layers=0,
            )

    def test_cv_type_in_dv_fails(self):
        with pytest.raises(ValueError, match="not valid for DV"):
            Ansatz(
                name="bad",
                paradigm=Paradigm.DV,
                n_sites=2,
                ansatz_type=AnsatzType.GAUSSIAN,
            )

    def test_dv_type_in_cv_fails(self):
        with pytest.raises(ValueError, match="not valid for CV"):
            Ansatz(
                name="bad",
                paradigm=Paradigm.CV,
                n_sites=2,
                ansatz_type=AnsatzType.HARDWARE_EFFICIENT,
            )

    def test_invalid_paradigm_type_fails(self):
        with pytest.raises(ValueError, match="Paradigm enum"):
            Ansatz(
                name="bad",
                paradigm="DV",
                n_sites=2,
                ansatz_type=AnsatzType.CUSTOM,
            )

    def test_invalid_ansatz_type_fails(self):
        with pytest.raises(ValueError, match="AnsatzType enum"):
            Ansatz(
                name="bad",
                paradigm=Paradigm.DV,
                n_sites=2,
                ansatz_type="hardware",
            )


class TestAnsatzUtils:

    def test_estimated_depth_with_entanglement(self):
        a = Ansatz(
            name="HEA",
            paradigm=Paradigm.DV,
            n_sites=4,
            ansatz_type=AnsatzType.HARDWARE_EFFICIENT,
            n_layers=3,
            entanglement_pattern="linear",
        )
        assert a.estimated_depth() == 33

    def test_estimated_depth_no_entanglement(self):
        a = Ansatz(
            name="local",
            paradigm=Paradigm.DV,
            n_sites=4,
            ansatz_type=AnsatzType.HARDWARE_EFFICIENT,
            n_layers=2,
        )
        assert a.estimated_depth() == 16

    def test_circuit_reference_optional(self):
        a = Ansatz(
            name="ref",
            paradigm=Paradigm.DV,
            n_sites=2,
            ansatz_type=AnsatzType.CUSTOM,
            circuit_reference="circuit-uuid-123",
        )
        assert a.circuit_reference == "circuit-uuid-123"

    def test_n_trainable_all_trainable(self):
        a = Ansatz(
            name="all-trainable",
            paradigm=Paradigm.DV,
            n_sites=2,
            ansatz_type=AnsatzType.CUSTOM,
            parameters=[
                Parameter(name="theta"),
                Parameter(name="phi"),
            ],
        )
        assert a.n_trainable() == 2

    def test_n_trainable_none_trainable(self):
        a = Ansatz(
            name="frozen",
            paradigm=Paradigm.DV,
            n_sites=2,
            ansatz_type=AnsatzType.CUSTOM,
            parameters=[
                Parameter(name="theta", trainable=False),
                Parameter(name="phi", trainable=False),
            ],
        )
        assert a.n_trainable() == 0

    def test_len_no_parameters(self):
        a = Ansatz(
            name="empty",
            paradigm=Paradigm.DV,
            n_sites=2,
            ansatz_type=AnsatzType.CUSTOM,
        )
        assert len(a) == 0

    def test_unique_id_by_instance(self):
        a1 = Ansatz(
            name="a", paradigm=Paradigm.DV,
            n_sites=2, ansatz_type=AnsatzType.CUSTOM
        )
        a2 = Ansatz(
            name="a", paradigm=Paradigm.DV,
            n_sites=2, ansatz_type=AnsatzType.CUSTOM
        )
        assert a1.id != a2.id

    def test_estimated_depth_full_entanglement(self):
        a = Ansatz(
            name="HEA-full",
            paradigm=Paradigm.DV,
            n_sites=4,
            ansatz_type=AnsatzType.HARDWARE_EFFICIENT,
            n_layers=2,
            entanglement_pattern="full",
        )
        # (2*4 + 6) * 2 = 28
        assert a.estimated_depth() == 28

    def test_estimated_depth_circular_entanglement(self):
        a = Ansatz(
            name="HEA-circular",
            paradigm=Paradigm.DV,
            n_sites=4,
            ansatz_type=AnsatzType.HARDWARE_EFFICIENT,
            n_layers=2,
            entanglement_pattern="circular",
        )
        # (2*4 + 4) * 2 = 24
        assert a.estimated_depth() == 24

    def test_native_gates_default_empty(self):
        a = Ansatz(
            name="no-gates",
            paradigm=Paradigm.DV,
            n_sites=2,
            ansatz_type=AnsatzType.CUSTOM,
        )
        assert a.native_gates == []

    def test_circuit_reference_none_by_default(self):
        a = Ansatz(
            name="no-ref",
            paradigm=Paradigm.DV,
            n_sites=2,
            ansatz_type=AnsatzType.CUSTOM,
        )
        assert a.circuit_reference is None

class TestAnsatzCVMethods:

    def test_estimated_depth_raises_for_cv(self):
        a = Ansatz(
            name="gaussian",
            paradigm=Paradigm.CV,
            n_sites=2,
            ansatz_type=AnsatzType.GAUSSIAN,
        )
        with pytest.raises(NotImplementedError):
            a.estimated_depth()

    def test_cv_complexity_raises_for_dv(self):
        a = Ansatz(
            name="hea",
            paradigm=Paradigm.DV,
            n_sites=2,
            ansatz_type=AnsatzType.HARDWARE_EFFICIENT,
        )
        with pytest.raises(ValueError):
            a.cv_complexity()

    def test_cv_complexity_gaussian_only(self):
        a = Ansatz(
            name="gaussian",
            paradigm=Paradigm.CV,
            n_sites=3,
            ansatz_type=AnsatzType.GAUSSIAN,
            n_layers=2,
        )
        result = a.cv_complexity()
        assert result["gaussian_ops"] == 6
        assert result["non_gaussian_ops"] == 0
        assert result["has_universal"] is False

    def test_cv_complexity_with_non_gaussian(self):
        a = Ansatz(
            name="non-gaussian",
            paradigm=Paradigm.CV,
            n_sites=2,
            ansatz_type=AnsatzType.NON_GAUSSIAN,
            n_layers=3,
            non_gaussian_ops_per_layer=1,
        )
        result = a.cv_complexity()
        assert result["non_gaussian_ops"] == 3
        assert result["has_universal"] is True

    def test_has_non_gaussian_false_for_gaussian_ansatz(self):
        a = Ansatz(
            name="gaussian",
            paradigm=Paradigm.CV,
            n_sites=2,
            ansatz_type=AnsatzType.GAUSSIAN,
        )
        assert a.has_non_gaussian() is False

    def test_has_non_gaussian_true_when_ops_present(self):
        a = Ansatz(
            name="non-gaussian",
            paradigm=Paradigm.CV,
            n_sites=2,
            ansatz_type=AnsatzType.NON_GAUSSIAN,
            non_gaussian_ops_per_layer=2,
        )
        assert a.has_non_gaussian() is True