from dataclasses import dataclass

from backend.domain.models.ansatz import Ansatz
from backend.domain.ports.ansatz_registry import AnsatzRegistry


@dataclass
class RegisterAnsatzRequest:
    ansatz: Ansatz
    overwrite: bool = False

@dataclass
class RegisterAnsatzResponse:
    ansatz: Ansatz
    was_overwritten: bool


class RegisterAnsatzUseCase:

    def __init__(self, registry: AnsatzRegistry):
        self._registry = registry

    def execute(self, request: RegisterAnsatzRequest,) -> RegisterAnsatzResponse:

        already_exists = self._registry.exists(request.ansatz.name)

        if already_exists and not request.overwrite:
            raise ValueError(
                f"Ansatz '{request.ansatz.name}' already exists. "
                f"Use overwrite=True to replace it."
            )

        if already_exists:
            self._registry.unregister(request.ansatz.name)

        self._registry.register(request.ansatz)

        return RegisterAnsatzResponse(
            ansatz=request.ansatz,
            was_overwritten=already_exists,
        )