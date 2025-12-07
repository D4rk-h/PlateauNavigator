from typing import List, Optional, Dict, Any


class UnitaryTransformation:
    def __init__(self, gate_name: str, qubits: List[int], params: Optional[Dict[str, float]] = None):
        self.gate_name = gate_name
        self.qubits = qubits
        self.params = params if params is not None else {}

    def __repr__(self):
        return f"Op:{self.gate_name}, qubits:{self.qubits}, params:{self.params if self.params else None} "