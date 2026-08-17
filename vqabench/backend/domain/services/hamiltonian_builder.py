from backend.domain.models.hamiltonian import Hamiltonian, PauliTerm, CVTerm, Paradigm
from backend.domain.models.physical_model import (
    PhysicalModel,
    BoseHubbardModel,
    KerrOscillatorModel,
    IsingModel,
    HeisenbergModel,
    FermiHubbardModel,
    MoleculeModel,
    BoundaryCondition,
)

class HamiltonianBuilderService:
    def build(self, model: PhysicalModel) -> Hamiltonian:
        if isinstance(model, BoseHubbardModel):
            return self._build_bose_hubbard_hamiltonian(model)
        if isinstance(model, KerrOscillatorModel):
            return self._build_kerr_oscillator_hamiltonian(model)
        if isinstance(model, IsingModel):
            return self._build_ising_hamiltonian(model)
        if isinstance(model, HeisenbergModel):
            return self._build_heisenberg_hamiltonian(model)
        if isinstance(model, FermiHubbardModel):
            return self._build_fermi_hubbard_hamiltonian(model)
        if isinstance(model, MoleculeModel):
            return self._build_molecule_hamiltonian(model)
        raise ValueError(
            f"No Hamiltonian builder registered for {type(model).__name__}. Implement a builder method and register it in build()."
        ) 

    def _build_bose_hubbard(self, model: BoseHubbardModel) -> Hamiltonian:
        terms: list[CVTerm] = []
        n = model.n_sites

        bonds = (
            [(i, (i + 1) % n) for i in range(n)]
            if model.boundary == BoundaryCondition.PERIODIC
            else [(i, i + 1) for i in range(n - 1)]
        )

        for i, j in bonds:
            terms.append(CVTerm(operator="a_dag", modes=[i, j], coefficient=-model.t))
            terms.append(CVTerm(operator="a", modes=[j, i], coefficient=-model.t))

        for i in range(n):
            terms.append(CVTerm(operator="n", modes=[i], coefficient=model.u / 2, power=2))
            terms.append(CVTerm(operator="n", modes=[i], coefficient=-model.u / 2))

        if model.mu != 0.0: 
            for i in range(n):
                terms.append(CVTerm(operator="n", modes=[i], coefficient=-model.mu))

        return Hamiltonian(
            name=f"BoseHubbard-{model.n_sites}-sites",
            paradigm=Paradigm.CV,
            n_sites=n,
            terms=terms,
        )

    def _build_kerr_oscillator(self, model: KerrOscillatorModel) -> Hamiltonian:
        terms = [
            CVTerm(operator="n", modes=[0], coefficient=model.omega),
            CVTerm(operator="a_dag", modes=[0, 0], coefficient=model.chi / 2, power=2),
        ]

        return Hamiltonian(
            name=f"KerrOscillator",
            paradigm=Paradigm.CV,
            n_sites=1,
            terms=terms,
        )

    def _build_ising(self, model: IsingModel) -> Hamiltonian:
        terms: list[PauliTerm] = []
        n = model.n_sites

        bonds = (
            [(i, (i + 1) % n) for i in range(n)]
            if model.boundary == BoundaryCondition.PERIODIC
            else [(i, i + 1) for i in range(n - 1)]
        )

        for i, j in bonds:
            pauli = self._pauli_string(n, {i: "Z", j: "Z"})
            terms.append(PauliTerm(pauli_string=pauli, coefficient=-model.j))

        for i in range(n):
            pauli = self._pauli_string(n, {i: "X"})
            terms.append(PauliTerm(pauli_string=pauli, coefficient=-model.h))

        return Hamiltonian(
            name=f"Ising-{model.n_sites}-sites",
            paradigm=Paradigm.DV,
            n_sites=n,
            terms=terms,
        )

    def _build_heisenberg(self, model: HeisenbergModel) -> Hamiltonian:
        terms: list[PauliTerm] = []
        n = model.n_sites

        bonds = (
            [(i, (i + 1) % n) for i in range(n)]
            if model.boundary == BoundaryCondition.PERIODIC
            else [(i, i + 1) for i in range(n - 1)]
        )

        for i, j in bonds:
            if model.Jx != 0.0:
                terms.append(PauliTerm(
                    pauli_string=self._pauli_string(n, {i: "X", j: "X"}),
                    coefficient=model.Jx,
                ))
            if model.Jy != 0.0:
                terms.append(PauliTerm(
                    pauli_string=self._pauli_string(n, {i: "Y", j: "Y"}),
                    coefficient=model.Jy,
                ))
            if model.Jz != 0.0:
                terms.append(PauliTerm(
                    pauli_string=self._pauli_string(n, {i: "Z", j: "Z"}),
                    coefficient=model.Jz,
                ))

        return Hamiltonian(
            name=f"Heisenberg-{model.model_subtype()}-{n}-sites",
            paradigm=Paradigm.DV,
            n_sites=n,
            terms=terms,
        )

    def _build_fermi_hubbard(self, model: FermiHubbardModel) -> Hamiltonian:
        n = model.n_sites
        n_qubits = model.n_qubits()
        terms: list[PauliTerm] = []

        bonds = (
            [(i, (i + 1) % n) for i in range(n)]
            if model.boundary == BoundaryCondition.PERIODIC
            else [(i, i + 1) for i in range(n - 1)]
        )

        for spin_offset in [0, n]:
            for i, j in bonds:
                qi, qj = i + spin_offset, j + spin_offset
                terms.append(PauliTerm(
                    pauli_string=self._pauli_string(n_qubits, {qi: "X", qj: "X"}),
                    coefficient=-model.t / 2,
                ))
                terms.append(PauliTerm(
                    pauli_string=self._pauli_string(n_qubits, {qi: "Y", qj: "Y"}),
                    coefficient=-model.t / 2,
                ))

        for i in range(n):
            qi_up, qi_down = i, i + n
            terms.append(PauliTerm(
                pauli_string=self._pauli_string(n_qubits, {}),
                coefficient=model.u / 4,
            ))
            terms.append(PauliTerm(
                pauli_string=self._pauli_string(n_qubits, {qi_up: "Z"}),
                coefficient=-model.u / 4,
            ))
            terms.append(PauliTerm(
                pauli_string=self._pauli_string(n_qubits, {qi_down: "Z"}),
                coefficient=-model.u / 4,
            ))
            terms.append(PauliTerm(
                pauli_string=self._pauli_string(n_qubits, {qi_up: "Z", qi_down: "Z"}),
                coefficient=model.u / 4,
            ))

        return Hamiltonian(
            name=f"FermiHubbard-{model.n_sites}-sites",
            paradigm=Paradigm.DV,
            n_sites=n_qubits,
            terms=terms,
        )

    def _build_molecule(self, model: MoleculeModel) -> Hamiltonian:
        """
        Molecular Hamiltonian placeholder, full version requires OpenFermion + PySCF integration.
        
        This will be replaced by the OpenFermion adapter in close future.
        """
        
        n = model.n_sites
        terms = [
            PauliTerm(pauli_string="I" * n, coefficient=0.0,)
        ]
        return Hamiltonian(
            name=f"{model.molecule_name}-{model.basis_set}-placeholder",
            paradigm=Paradigm.DV,
            n_sites=n,
            terms=terms,
        )

    def _pauli_string(
            self,
            n_sites: int,
            operators: dict[int, str],
    ) -> str:
        result = ["I"] * n_sites
        for idx, op in operators.items():
            result[idx] = op
        return "".join(result)
    
