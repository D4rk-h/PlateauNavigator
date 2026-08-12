import pytest
from backend.domain.models.circuit import Circuit, CircuitSourceType
from backend.domain.models.hamiltonian import Paradigm
from backend.domain.models.parameter import Parameter


class TestCircuitCreation:

    def test_create_dv_qasm3_circuit(self):
        c = Circuit(
            name="Hardware-Efficient",
            paradigm=Paradigm.DV,
            n_sites=4,
            source='OPENQASM 3.0;\nqubit[4] q;\nry(theta[0]) q[0];',
            source_type=CircuitSourceType.QASM3,
        )
        assert c.name == "Hardware-Efficient"
        assert c.paradigm == Paradigm.DV
        assert c.n_sites == 4
        assert c.source_type == CircuitSourceType.QASM3
        assert c.is_parameterized() is False

    def test_create_dv_qiskit_circuit(self):
        c = Circuit(
            name="UCCSD",
            paradigm=Paradigm.DV,
            n_sites=2,
            source="from qiskit import QuantumCircuit\nqc = QuantumCircuit(2)",
            source_type=CircuitSourceType.QISKIT,
        )
        assert c.source_type == CircuitSourceType.QISKIT

    def test_create_cv_python_circuit(self):
        c = Circuit(
            name="Gaussian-Ansatz",
            paradigm=Paradigm.CV,
            n_sites=2,
            source="def ansatz(n_modes, params):\n    pass",
            source_type=CircuitSourceType.PYTHON_MRMUSTARD,
        )
        assert c.paradigm == Paradigm.CV
        assert c.source_type == CircuitSourceType.PYTHON_MRMUSTARD

    def test_circuit_with_parameters(self):
        c = Circuit(
            name="Parameterized",
            paradigm=Paradigm.DV,
            n_sites=2,
            source="ry(theta) q[0];",
            source_type=CircuitSourceType.QASM3,
            parameters=[
                Parameter(name="theta", value=0.5),
                Parameter(name="phi", value=0.0, trainable=False),
            ],
        )
        assert len(c) == 2
        assert c.is_parameterized() is True
        assert c.n_trainable() == 1
        assert c.trainable_parameters()[0].name == "theta"


class TestCircuitValidations:

    def test_zero_n_sites_fails(self):
        with pytest.raises(ValueError, match="positive integer"):
            Circuit(
                name="bad",
                paradigm=Paradigm.DV,
                n_sites=0,
                source="x q[0];",
                source_type=CircuitSourceType.QASM3,
            )

    def test_empty_source_fails(self):
        with pytest.raises(ValueError, match="non-empty string"):
            Circuit(
                name="bad",
                paradigm=Paradigm.DV,
                n_sites=2,
                source=" ",
                source_type=CircuitSourceType.QASM3,
            )

    def test_cv_with_qasm3_source_fails(self):
        with pytest.raises(ValueError, match="PYTHON_MRMUSTARD"):
            Circuit(
                name="bad",
                paradigm=Paradigm.CV,
                n_sites=2,
                source="OPENQASM 3.0;",
                source_type=CircuitSourceType.QASM3,
            )

    def test_dv_with_python_source_fails(self):
        with pytest.raises(ValueError, match="QISKIT"):
            Circuit(
                name="bad",
                paradigm=Paradigm.DV,
                n_sites=2,
                source="def ansatz(): pass",
                source_type=CircuitSourceType.PYTHON_MRMUSTARD,
            )

    def test_invalid_paradigm_type_fails(self):
        with pytest.raises(ValueError, match="Paradigm Enum"):
            Circuit(
                name="bad",
                paradigm="DV",
                n_sites=2,
                source="x q[0];",
                source_type=CircuitSourceType.QASM3,
            )


class TestCircuitUtils:

    def test_trainable_parameters_filter(self):
        c = Circuit(
            name="test",
            paradigm=Paradigm.DV,
            n_sites=2,
            source="ry(theta) q[0];",
            source_type=CircuitSourceType.QASM3,
            parameters=[
                Parameter(name="a", value=1.0, trainable=True),
                Parameter(name="b", value=2.0, trainable=False),
                Parameter(name="c", value=3.0, trainable=True),
            ],
        )
        trainable = c.trainable_parameters()
        assert len(trainable) == 2
        assert all(p.trainable for p in trainable)

    def test_description_optional(self):
        c = Circuit(
            name="test",
            paradigm=Paradigm.DV,
            n_sites=2,
            source="x q[0];",
            source_type=CircuitSourceType.QASM3,
            description="A simple test circuit",
        )
        assert c.description == "A simple test circuit"