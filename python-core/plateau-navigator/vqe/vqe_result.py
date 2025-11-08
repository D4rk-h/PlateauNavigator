from dataclasses import field, dataclass
from typing import List, Optional, Any, Dict

import numpy as np


@dataclass
class VQEResult:

    optimal_energy: float
    optimal_params: np.ndarray
    energy_history: np.ndarray
    param_history: np.ndarray
    gradient_history: Optional[np.ndarray] = None
    iterations: int = 0
    success: bool = False
    message: str = ""
    execution_time: float = 0.0
    energy_evaluations: int = 0
    gradient_evaluations: int = 0
    plateau_detected: bool = False
    plateau_iterations: List[int] = field(default_factory=list)
    gradient_variance: Optional[np.ndarray] = None
    backend_name: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            'optimal_energy': float(self.optimal_energy),
            'optimal_params': self.optimal_params.tolist(),
            'energy_history': self.energy_history.tolist(),
            'iterations': self.iterations,
            'success': self.success,
            'execution_time': self.execution_time,
            'plateau_detected': self.plateau_detected,
            'backends': self.backend_name
        }