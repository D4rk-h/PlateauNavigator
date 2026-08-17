import pytest
from backend.domain.models.physical_model import (
    BoseHubbardModel,
    KerrOscillatorModel,
    IsingModel,
    HeisenbergModel,
    FermiHubbardModel,
    MoleculeModel,
    BoundaryCondition,
    LatticeGeometry,
    BosonicModel,
    FermionicModel,
    PhysicalModel,
)
from backend.domain.models.hamiltonian import Paradigm


ALL_MODELS = [
    BoseHubbardModel(name="BH", n_sites=4),
    KerrOscillatorModel(name="Kerr"),
    IsingModel(name="Ising", n_sites=4),
    HeisenbergModel(name="Heis", n_sites=4),
    FermiHubbardModel(name="FH", n_sites=4),
    MoleculeModel(name="H2"),
]

class TestPhysicalModelContract:

    @pytest.mark.parametrize("model", ALL_MODELS)
    def test_base_contract(self, model):
        assert isinstance(model, PhysicalModel)
        assert model.id != ""
        summary = model.summary()
        assert "id" in summary
        assert "name" in summary
        assert "model" in summary
        assert "n_sites" in summary
        assert "preferred_paradigm" in summary
        assert "n_interactions" in summary

        assert isinstance(model.preferred_paradigm(), Paradigm)
        assert isinstance(model.n_interactions(), int)
        assert model.n_interactions() > 0
        
    def test_zero_n_sites_fails_for_all_models(self):
        with pytest.raises(ValueError):
            IsingModel(name="bad", n_sites=0)
        with pytest.raises(ValueError):
            HeisenbergModel(name="bad", n_sites=0)
        with pytest.raises(ValueError):
            FermiHubbardModel(name="bad", n_sites=0)
        with pytest.raises(ValueError):
            BoseHubbardModel(name="bad", n_sites=0)

class TestBoseHubbardModel:

    def test_default_creation(self):
        m = BoseHubbardModel(name="BH", n_sites=4)
        assert m.n_sites == 4
        assert m.t == 1.0
        assert m.u == 1.0
        assert m.mu == 0.0
        assert m.boundary == BoundaryCondition.OPEN
        assert m.max_occupation == 2

    def test_is_bosonic_model(self):
        m = BoseHubbardModel(name="BH", n_sites=4)
        assert isinstance(m, BosonicModel)

    def test_preferred_paradigm_is_cv(self):
        m = BoseHubbardModel(name="BH", n_sites=4)
        assert m.preferred_paradigm() == Paradigm.CV

    def test_n_interactions_open(self):
        m = BoseHubbardModel(name="BH", n_sites=4)
        assert m.n_interactions() == 3

    def test_n_interactions_periodic(self):
        m = BoseHubbardModel(
            name="BH", n_sites=4,
            boundary=BoundaryCondition.PERIODIC
        )
        assert m.n_interactions() == 4

    def test_negative_U_fails(self):
        with pytest.raises(ValueError, match="non-negative"):
            BoseHubbardModel(name="BH", n_sites=4, u=-1.0)

    def test_zero_max_occupation_fails(self):
        with pytest.raises(ValueError, match="max_occupation"):
            BoseHubbardModel(name="BH", n_sites=4, max_occupation=0)

    def test_superfluid_regime_true(self):
        m = BoseHubbardModel(name="BH", n_sites=4, t=10.0, u=1.0)
        assert m.is_superfluid() is True

    def test_superfluid_regime_false(self):
        m = BoseHubbardModel(name="BH", n_sites=4, t=0.1, u=10.0)
        assert m.is_superfluid() is False

    def test_mott_insulator_regime_true(self):
        m = BoseHubbardModel(name="BH", n_sites=4, t=0.01, u=100.0)
        assert m.is_mott_insulator() is True

    def test_mott_insulator_regime_false(self):
        m = BoseHubbardModel(name="BH", n_sites=4, t=1.0, u=1.0)
        assert m.is_mott_insulator() is False

    def test_summary_has_model_specific_fields(self):
        m = BoseHubbardModel(name="BH", n_sites=4)
        summary = m.summary()
        assert "t" in summary
        assert "u" in summary
        assert "mu" in summary
        assert "boundary" in summary
        assert "max_occupation" in summary

    def test_unique_ids(self):
        m1 = BoseHubbardModel(name="BH", n_sites=4)
        m2 = BoseHubbardModel(name="BH", n_sites=4)
        assert m1.id != m2.id


class TestKerrOscillatorModel:

    def test_default_creation(self):
        m = KerrOscillatorModel(name="Kerr")
        assert m.n_sites == 1
        assert m.omega == 1.0
        assert m.chi == 0.1

    def test_is_bosonic_model(self):
        m = KerrOscillatorModel(name="Kerr")
        assert isinstance(m, BosonicModel)

    def test_preferred_paradigm_is_cv(self):
        m = KerrOscillatorModel(name="Kerr")
        assert m.preferred_paradigm() == Paradigm.CV

    def test_n_sites_always_one(self):
        m = KerrOscillatorModel(name="Kerr")
        assert m.n_sites == 1

    def test_n_interactions_is_one(self):
        m = KerrOscillatorModel(name="Kerr")
        assert m.n_interactions() == 1

    def test_is_non_gaussian_true_when_chi_nonzero(self):
        m = KerrOscillatorModel(name="Kerr", chi=0.5)
        assert m.is_non_gaussian() is True

    def test_is_non_gaussian_false_when_chi_zero(self):
        m = KerrOscillatorModel(name="Kerr", chi=0.0)
        assert m.is_non_gaussian() is False

    def test_negative_omega_fails(self):
        with pytest.raises(ValueError, match="positive"):
            KerrOscillatorModel(name="Kerr", omega=-1.0)

    def test_zero_omega_fails(self):
        with pytest.raises(ValueError, match="positive"):
            KerrOscillatorModel(name="Kerr", omega=0.0)

    def test_summary_has_model_specific_fields(self):
        m = KerrOscillatorModel(name="Kerr")
        summary = m.summary()
        assert "omega" in summary
        assert "chi" in summary
        assert "non_gaussian" in summary


class TestIsingModel:

    def test_default_creation(self):
        m = IsingModel(name="Ising", n_sites=4)
        assert m.j == 1.0
        assert m.h == 0.5
        assert m.boundary == BoundaryCondition.OPEN
        assert m.geometry == LatticeGeometry.CHAIN

    def test_is_fermionic_model(self):
        m = IsingModel(name="Ising", n_sites=4)
        assert isinstance(m, FermionicModel)

    def test_preferred_paradigm_is_dv(self):
        m = IsingModel(name="Ising", n_sites=4)
        assert m.preferred_paradigm() == Paradigm.DV

    def test_n_interactions_open_chain(self):
        m = IsingModel(name="Ising", n_sites=4)
        assert m.n_interactions() == 3

    def test_n_interactions_periodic_chain(self):
        m = IsingModel(
            name="Ising", n_sites=4,
            boundary=BoundaryCondition.PERIODIC
        )
        assert m.n_interactions() == 4

    def test_n_interactions_square_2x2_open(self):
        m = IsingModel(
            name="Ising", n_sites=4,
            geometry=LatticeGeometry.SQUARE,
            boundary=BoundaryCondition.OPEN
        )
        assert m.n_interactions() == 4

    def test_n_interactions_square_2x2_periodic(self):
        m = IsingModel(
            name="Ising", n_sites=4,
            geometry=LatticeGeometry.SQUARE,
            boundary=BoundaryCondition.PERIODIC
        )
        assert m.n_interactions() == 6

    def test_square_non_perfect_square_fails(self):
        with pytest.raises(ValueError, match="perfect square"):
            IsingModel(
                name="Ising", n_sites=3,
                geometry=LatticeGeometry.SQUARE
            )

    def test_is_critical_true(self):
        m = IsingModel(name="Ising", n_sites=4, j=1.0, h=1.0)
        assert m.is_critical() is True

    def test_is_critical_false(self):
        m = IsingModel(name="Ising", n_sites=4, j=1.0, h=0.5)
        assert m.is_critical() is False

    def test_is_critical_works_with_negative_j(self):
        m = IsingModel(name="Ising", n_sites=4, j=-1.0, h=1.0)
        assert m.is_critical() is True

    def test_summary_has_model_specific_fields(self):
        m = IsingModel(name="Ising", n_sites=4)
        summary = m.summary()
        assert "j" in summary
        assert "h" in summary
        assert "boundary" in summary
        assert "geometry" in summary
        assert "critical" in summary


class TestHeisenbergModel:

    def test_default_is_xxx(self):
        m = HeisenbergModel(name="Heis", n_sites=4)
        assert m.model_subtype() == "XXX"

    def test_xxz_subtype(self):
        m = HeisenbergModel(name="Heis", n_sites=4, Jz=0.5)
        assert m.model_subtype() == "XXZ"

    def test_xyz_subtype(self):
        m = HeisenbergModel(name="Heis", n_sites=4, Jx=1.0, Jy=0.5, Jz=0.3)
        assert m.model_subtype() == "XYZ"

    def test_is_fermionic_model(self):
        m = HeisenbergModel(name="Heis", n_sites=4)
        assert isinstance(m, FermionicModel)

    def test_preferred_paradigm_is_dv(self):
        m = HeisenbergModel(name="Heis", n_sites=4)
        assert m.preferred_paradigm() == Paradigm.DV

    def test_n_interactions_open(self):
        m = HeisenbergModel(name="Heis", n_sites=4)
        assert m.n_interactions() == 3

    def test_n_interactions_periodic(self):
        m = HeisenbergModel(
            name="Heis", n_sites=4,
            boundary=BoundaryCondition.PERIODIC
        )
        assert m.n_interactions() == 4

    def test_summary_has_model_specific_fields(self):
        m = HeisenbergModel(name="Heis", n_sites=4)
        summary = m.summary()
        assert "Jx" in summary
        assert "Jy" in summary
        assert "Jz" in summary
        assert "subtype" in summary
        assert "boundary" in summary

    def test_negative_couplings_valid(self):
        m = HeisenbergModel(name="Heis", n_sites=4, Jx=-1.0, Jy=-1.0, Jz=-1.0)
        assert m.model_subtype() == "XXX"


class TestFermiHubbardModel:

    def test_default_creation(self):
        m = FermiHubbardModel(name="FH", n_sites=4)
        assert m.t == 1.0
        assert m.u == 2.0
        assert m.mu == 0.0

    def test_is_fermionic_model(self):
        m = FermiHubbardModel(name="FH", n_sites=4)
        assert isinstance(m, FermionicModel)

    def test_preferred_paradigm_is_dv(self):
        m = FermiHubbardModel(name="FH", n_sites=4)
        assert m.preferred_paradigm() == Paradigm.DV

    def test_n_qubits_is_twice_n_sites(self):
        m = FermiHubbardModel(name="FH", n_sites=4)
        assert m.n_qubits() == 8

    def test_n_interactions_open(self):
        m = FermiHubbardModel(name="FH", n_sites=4)
        assert m.n_interactions() == 10

    def test_n_interactions_periodic(self):
        m = FermiHubbardModel(
            name="FH", n_sites=4,
            boundary=BoundaryCondition.PERIODIC
        )
        assert m.n_interactions() == 12

    def test_single_site_fails(self):
        with pytest.raises(ValueError, match="at least 2"):
            FermiHubbardModel(name="FH", n_sites=1)

    def test_is_half_filling_true(self):
        m = FermiHubbardModel(name="FH", n_sites=4, u=2.0, mu=1.0)
        assert m.is_half_filling() is True

    def test_is_half_filling_false(self):
        m = FermiHubbardModel(name="FH", n_sites=4, u=2.0, mu=0.0)
        assert m.is_half_filling() is False

    def test_summary_has_model_specific_fields(self):
        m = FermiHubbardModel(name="FH", n_sites=4)
        summary = m.summary()
        assert "t" in summary
        assert "u" in summary
        assert "mu" in summary
        assert "n_qubits" in summary
        assert "boundary" in summary
        assert "half_filling" in summary


class TestMoleculeModel:

    def test_h2_sets_n_sites_automatically(self):
        m = MoleculeModel(name="H2")
        assert m.n_sites == 4

    def test_lih_sets_n_sites_automatically(self):
        m = MoleculeModel(name="LiH mol", molecule_name="LiH")
        assert m.n_sites == 12

    def test_h2o_sets_n_sites_automatically(self):
        m = MoleculeModel(name="H2O mol", molecule_name="H2O")
        assert m.n_sites == 14

    def test_unknown_molecule_with_explicit_n_sites(self):
        m = MoleculeModel(name="custom", molecule_name="CustomMol", n_sites=8)
        assert m.n_sites == 8

    def test_unknown_molecule_without_n_sites_fails(self):
        with pytest.raises(ValueError):
            MoleculeModel(name="custom", molecule_name="CustomMol")

    def test_preferred_paradigm_is_dv(self):
        m = MoleculeModel(name="H2")
        assert m.preferred_paradigm() == Paradigm.DV

    def test_is_known_true_for_h2(self):
        m = MoleculeModel(name="H2")
        assert m.is_known() is True

    def test_is_known_false_for_custom(self):
        m = MoleculeModel(name="custom", molecule_name="CustomMol", n_sites=8)
        assert m.is_known() is False

    def test_n_electrons_h2(self):
        m = MoleculeModel(name="H2")
        assert m.n_electrons() == 2

    def test_n_electrons_lih(self):
        m = MoleculeModel(name="LiH", molecule_name="LiH")
        assert m.n_electrons() == 4

    def test_invalid_charge_fails(self):
        with pytest.raises(ValueError, match="charge"):
            MoleculeModel(name="H2", charge=5)

    def test_invalid_multiplicity_fails(self):
        with pytest.raises(ValueError, match="multiplicity"):
            MoleculeModel(name="H2", multiplicity=0)

    def test_negative_bond_length_fails(self):
        with pytest.raises(ValueError, match="bond_length"):
            MoleculeModel(name="H2", bond_length=-1.0)

    def test_none_bond_length_is_valid(self):
        m = MoleculeModel(name="H2")
        assert m.bond_length is None

    def test_summary_has_model_specific_fields(self):
        m = MoleculeModel(name="H2")
        summary = m.summary()
        assert "molecule" in summary
        assert "basis_set" in summary
        assert "charge" in summary
        assert "multiplicity" in summary
        assert "bond_length" in summary
        assert "n_electrons" in summary
        assert "known" in summary