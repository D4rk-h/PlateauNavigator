from typing import List
from ansatz_to_qasm import AnsatzToQasm


class MockBackend():
    def __init__(self, qasm_converter: AnsatzToQasm):
        self.qasm_converter = qasm_converter
        self.name = "QasmRecorder"

    def create_circuit(self, num_qubits: int):
        return {"status": "recorded"}

    def add_gate(self, gate_type: str, qubits: List[int], **params):
        self.qasm_converter.add_operation(gate_type, qubits, **params)
        return {"status": "recorded", "gate": gate_type}

    def execute_circuit(self):
        return {"status": "recorded"}

    def clear_circuit(self):
        self.qasm_converter.reset()