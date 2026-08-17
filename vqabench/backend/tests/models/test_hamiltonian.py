import pytest
from backend.domain.models.hamiltonian import (
    Hamiltonian,
    PauliTerm,
    CVTerm,
    Paradigm,
)
from backend.domain.models.parameter import Parameter


class TestPauliTerm:

    def test_create_basic_pauli_term(self):
        term = PauliTerm(pauli_string="ZZII", coefficient=1.0)
        assert term.pauli_string == "ZZII"
        assert term.coefficient == 1.0

    def test_empty_pauli_string_fails(self):
        with pytest.raises(ValueError, match="cannot be empty"):
            PauliTerm(pauli_string="")

    def test_invalid_char_pauli_string_fails(self):
        with pytest.raises(ValueError, match="invalid characters"):
            PauliTerm(pauli_string="ZABC")

    def test_permitted_zero_coefficient(self):
        term = PauliTerm(pauli_string="XIXI", coefficient=0.0)
        assert term.coefficient == 0.0

class TestCVTerm:
    def test_create_basic_cv_term(self):
        t = CVTerm(operator="n", modes=[0, 1], coefficient=0.5)
        assert t.operator == "n"
        assert t.modes == [0, 1]
        assert t.coefficient == 0.5

    def test_empty_modes_fails(self):
        with pytest.raises(ValueError, match="cannot be empty"):
            CVTerm(operator="x", modes=[])

    def test_negative_mode_fails(self):
        with pytest.raises(ValueError, match="non-negative"):
            CVTerm(operator="x", modes=[-1])

    def test_invalid_power_fails(self):
        with pytest.raises(ValueError, match="positive integer"):
            CVTerm(operator="x", modes=[0], power=0)

    def test_invalid_operator_fails(self):
        with pytest.raises(ValueError, match="must be one of"):
            CVTerm(operator="h", modes=[0])

class TestHamiltonianDV:

    def test_create_basic_dv_hamiltonian(self):
        h = Hamiltonian(
            name="Ising-1D",
            paradigm=Paradigm.DV,
            n_sites=4,
            terms=[
                PauliTerm("ZZII", 1.0),
                PauliTerm("IZZI", 1.0),
                PauliTerm("IIZZ", 1.0),
            ],
        )
        assert h.name == "Ising-1D"
        assert h.paradigm == Paradigm.DV
        assert h.n_sites == 4
        assert len(h) == 3
        assert h.total_coefficient_norm() == 3.0

    def test_inconsistent_pauli_string_length_fails(self):
        with pytest.raises(ValueError, match="length"):
            Hamiltonian(
                name="bad",
                paradigm=Paradigm.DV,
                n_sites=4,
                terms=[PauliTerm("ZZZ", 1.0)],
            )

    def test_cv_term_in_dv_hamiltonian_fails(self):
        with pytest.raises(ValueError, match="All terms must be PauliTerm"):
            Hamiltonian(
                name="bad",
                paradigm=Paradigm.DV,
                n_sites=2,
                terms=[CVTerm(operator="n", modes=[0])],
            )

    def test_hamiltonian_with_parameters(self):
        h = Hamiltonian(
            name="Parametric-Heisenberg",
            paradigm=Paradigm.DV,
            n_sites=2,
            terms=[PauliTerm("XX", 1.0), PauliTerm("YY", 1.0)],
            parameters=[Parameter(name="J", value=1.0)],
        )
        assert h.is_parameterized() is True
        assert h.parameters[0].name == "J"

    def test_zero_n_sites_fails(self):
        with pytest.raises(ValueError, match="positive integer"):
            Hamiltonian(name="bad", paradigm=Paradigm.DV, n_sites=0, terms=[PauliTerm("I", 1.0)])

class TestHamiltonianCV:

    def test_create_basic_cv_hamiltonian(self):
        h = Hamiltonian(
            name="Kerr-oscillator",
            paradigm=Paradigm.CV,
            n_sites=2,
            terms=[
                CVTerm(operator="n", modes=[0], coefficient=1.0),
                CVTerm(operator="n", modes=[1], coefficient=1.0),
                CVTerm(operator="a_dag", modes=[0, 1], coefficient=0.5),
            ],
        )
        assert h.paradigm == Paradigm.CV
        assert len(h) == 3

    def test_mode_out_of_range_fails(self):
        with pytest.raises(ValueError, match="out of range"):
            Hamiltonian(
                name="bad",
                paradigm=Paradigm.CV,
                n_sites=2,
                terms=[CVTerm(operator="n", modes=[2])],
            )

    def test_pauli_term_in_cv_hamiltonian_fails(self):
        with pytest.raises(ValueError, match="All terms must be CVTerm"):
            Hamiltonian(
                name="bad",
                paradigm=Paradigm.CV,
                n_sites=2,
                terms=[PauliTerm("ZI", 1.0)],
            )


class TestHamiltonianUtils:

    def test_total_coefficient_norm(self):
        h = Hamiltonian(
            name="test",
            paradigm=Paradigm.DV,
            n_sites=2,
            terms=[
                PauliTerm("ZI", 2.0),
                PauliTerm("IZ", -3.0),
            ],
        )
        assert h.total_coefficient_norm() == 5.0

    def test_is_parameterized_false(self):
        h = Hamiltonian(
            name="test",
            paradigm=Paradigm.DV,
            n_sites=2,
            terms=[PauliTerm("ZI", 1.0)],
        )
        assert h.is_parameterized() is False