from dataclasses import dataclass, field
from typing import Optional
import uuid

@dataclass
class Parameter:
    name: str
    value: float = 0.0
    trainable: bool = True
    lower_bound: Optional[float] = None
    upper_bound: Optional[float] = None
    id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def __post_init__(self):
        if self.lower_bound is not None and self.upper_bound is not None:
            if self.lower_bound >= self.upper_bound:
                raise ValueError(
                    f"lower_bound {self.lower_bound} must be lower than upper_bound {self.upper_bound}"
                )
        if self.lower_bound is not None and self.value < self.lower_bound:
            raise ValueError(
                f"value {self.value} can not be lower than lower_bound {self.lower_bound}"
            )
        if self.upper_bound is not None and self.value > self.upper_bound:
            raise ValueError(
                f"value {self.value} can not be greater than upper_bound {self.upper_bound}"
            )

@dataclass
class ParameterVector:
    name: str
    size: int
    initial_values: list[float] = field(default_factory=list)

    def __post_init__(self):
        if self.initial_values and len(self.initial_values) != self.size:
            raise ValueError(
                f"initial_values have {len(self.initial_values)} elements but size {self.size}"
            )
        if not self.initial_values:
            self.initial_values = [0.0] * self.size

    def to_parameters(self) -> list[Parameter]:
        return [
            Parameter(
                name=f"{self.name}[{i}]",
                value=self.initial_values[i]
            )
            for i in range(self.size)
        ]

    def __len__(self) -> int:
        return self.size