from mock_backend import MockBackend
from typing import List, Callable

import numpy as np

from unitary_transformation import UnitaryTransformation

class AnsatzToQasm:
    def __init__(self, num_qubits: int, transformations: List[UnitaryTransformation],
                 num_bits: int, shots: int, parameter_names: List[str]):
        self.num_qubits = num_qubits
        self.transformations = transformations
        self.num_bits = num_bits
        self.shots = 1 if shots <= 0 else shots
        self.parameter_names = parameter_names

    def reset(self):
        self.transformations.clear()
        self.parameter_names.clear()

    def add_transformation(self, gate_type: str, qubits: List[int], **params):
        ut = UnitaryTransformation(gate_type, qubits, params)
        self.transformations.append(ut)

    def to_qasm(self, include_measurements: bool = True):
        qasm_lines = [
            "OPENQASM 2.0;",
            'include "qelib1.inc";',
            "",
            f"qreg q[{self.num_qubits}];",
        ]

        if include_measurements:
            qasm_lines.append(f"creg c[{self.num_bits}];")

        qasm_lines.append("")

        for ut in self.transformations:
            qasm_ut = self._unitary_to_qasm(ut)
            if qasm_ut:
                qasm_lines.append(qasm_ut)

        if include_measurements:
            qasm_lines.append("")
            qasm_lines.append("// Measurements: ")
            for bit in range(self.num_bits):
                qasm_lines.append(f"measure q[{bit}] -> c[{bit}];")

        return "\n".join(qasm_lines)

    @staticmethod
    def _unitary_to_qasm(transformation: UnitaryTransformation) -> str:
        gate = transformation.gate_name.lower()
        qubits = transformation.qubits
        params = transformation.params

        if gate in ["h", "hadamard"]:
            return f"h q[{qubits[0]}];"

        elif gate in ["x", "pauli-x"]:
            return f"x q[{qubits[0]}];"

        elif gate in ["y", "pauli-y"]:
            return f"y q[{qubits[0]}];"

        elif gate in ["z", "pauli-z"]:
            return f"z q[{qubits[0]}];"

        elif gate in ["s", "s-gate"]:
            return f"s q[{qubits[0]}];"

        elif gate in ["t", "t-gate"]:
            return f"t q[{qubits[0]}];"

        elif gate == "sdg":
            return f"sdg q[{qubits[0]}];"

        elif gate == "tdg":
            return f"tdg q[{qubits[0]}];"

        elif gate in ["rx"]:
            theta = params.get("theta", 0.0)
            return f"rx({theta}) q[{qubits[0]}];"

        elif gate in ["ry"]:
            theta = params.get("theta", 0.0)
            return f"ry({theta}) q[{qubits[0]}];"

        elif gate in ["rz"]:
            phi = params.get("phi", 0.0)
            return f"rz({phi}) q[{qubits[0]}];"

        elif gate in ["u", "unitary"]:
            theta = params.get("theta", 0.0)
            phi = params.get("phi", 0.0)
            lam = params.get("lambda", 0.0)
            return f"u3({theta},{phi},{lam}) q[{qubits[0]}];"

        elif gate in ["cx", "cnot"]:
            return f"cx q[{qubits[0]}],q[{qubits[1]}];"

        elif gate in ["cy"]:
            return f"cy q[{qubits[0]}],q[{qubits[1]}];"

        elif gate in ["cz"]:
            return f"cz q[{qubits[0]}],q[{qubits[1]}];"

        elif gate in ["swap"]:
            return f"swap q[{qubits[0]}],q[{qubits[1]}];"

        elif gate in ["ccx", "toffoli"]:
            return f"ccx q[{qubits[0]}],q[{qubits[1]}],q[{qubits[2]}];"

        else:
            raise ValueError(f"Unsupported gate type for QASM conversion: {gate}")

    def from_ansatz_function(self, ansatz_fn: Callable, params: np.ndarray) -> str:
        mock_backend = MockBackend(self)
        self.reset()
        ansatz_fn(mock_backend, params)
        return self.to_qasm(include_measurements=True)

def convert_ansatz_to_qasm(ansatz_function: Callable, params: np.ndarray, num_qubits: int) -> str:
    converter = AnsatzToQasm(num_qubits)
    return converter.from_ansatz_function(ansatz_function, params)