from typing import Dict, Tuple, List


_GATE_QASM: Dict[str, Tuple[str, int, List[str]]] = {
    "h":        ("h",    1, []),
    "x":        ("x",    1, []),
    "y":        ("y",    1, []),
    "z":        ("z",    1, []),
    "s":        ("s",    1, []),
    "t":        ("t",    1, []),
    "rx":       ("rx",   1, ["theta"]),
    "ry":       ("ry",   1, ["theta"]),
    "rz":       ("rz",   1, ["phi"]),
    "cx":       ("cx",   2, []),
    "cnot":     ("cx",   2, []),
    "cz":       ("cz",   2, []),
    "swap":     ("swap", 2, []),
    "ccx":      ("ccx",  3, []),
    "toffoli":  ("ccx",  3, []),
}

def _build_qasm(num_qubits: int, operations: List[Dict]) -> str:
    """
    Serialize a gate queue to OpenQASM 2.0.

    Measurements are NOT included — the Estimator primitive does not
    require explicit measurement gates. Adding them would conflict with
    mid-circuit operations and is incorrect for expectation value jobs.
    """
    lines = [
        'OPENQASM 2.0;',
        'include "qelib1.inc";',
        f'qreg q[{num_qubits}];',
    ]
    for op in operations:
        gate_key = op["gate_type"].lower()
        if gate_key not in _GATE_QASM:
            raise ValueError(
                f"Gate '{op['gate_type']}' not supported. "
                f"Supported: {sorted(_GATE_QASM.keys())}"
            )
        qasm_name, _, param_names = _GATE_QASM[gate_key]
        qubits_str = ", ".join(f"q[{q}]" for q in op["qubits"])
        if param_names:
            vals = [str(float(op["params"].get(p, 0.0))) for p in param_names]
            lines.append(f"{qasm_name}({', '.join(vals)}) {qubits_str};")
        else:
            lines.append(f"{qasm_name} {qubits_str};")
    return "\n".join(lines)
