from time import time
import numpy as np
from scipy.optimize import minimize
from typing import Callable, Dict, Any, Optional, Tuple
from .optimizer_type import OptimizerType
from .vqe_result import VQEResult


class VQE:
        def __init__(
                self,
                backend,
                hamiltonian: np.ndarray,
                ansatz: Callable,
                gradient_method: str = "parameter_shift",
                plateau_threshold: float = 1e-6,
                verbose: bool = True
        ):
            self.backend = backend
            self.hamiltonian = hamiltonian
            self.ansatz = ansatz
            self.gradient_method = gradient_method
            self.plateau_threshold = plateau_threshold
            self.verbose = verbose
            self._validate_hamiltonian()
            self.energy_history = []
            self.param_history = []
            self.gradient_history = []
            self.iteration = 0
            self.energy_eval_count = 0
            self.gradient_eval_count = 0
            self.plateau_iterations = []
            self.gradient_variances = []
            self.start_time = None

        def _validate_hamiltonian(self):
            if not isinstance(self.hamiltonian, np.ndarray):
                raise TypeError("Hamiltonian must be numpy array")

            if self.hamiltonian.ndim != 2:
                raise ValueError(f"Hamiltonian must be 2D matrix, got shape {self.hamiltonian.shape}")

            if self.hamiltonian.shape[0] != self.hamiltonian.shape[1]:
                raise ValueError(f"Hamiltonian must be square, got shape {self.hamiltonian.shape}")

            if not np.allclose(self.hamiltonian, self.hamiltonian.conj().T):
                raise ValueError("Hamiltonian must be Hermitian (H = H†)")

        def run(
                self,
                initial_params: np.ndarray,
                optimizer: OptimizerType = OptimizerType.COBYLA,
                max_iter: int = 1000,
                tol: float = 1e-6,
                callback: Optional[Callable] = None
            ) -> VQEResult:

            self.start_time = time()
            self._reset_tracking()

            if self.verbose:
                self._print_header(initial_params, optimizer)

            def objective(params: np.ndarray) -> float:
                energy = self._evaluate_energy(params)
                self._track_iteration(params, energy)
                if callback:
                    grads = self._last_gradients if hasattr(self, "_last_gradients") else None
                    callback(self.iteration, energy, params, grads)
                if self.verbose and self.iteration % 10 == 0:
                    self._print_progress()
                return energy

            jac = None
            if optimizer in [OptimizerType.BFGS, OptimizerType.L_BFGS_B, OptimizerType.SLSQP]:
                if self.gradient_method != "none":
                    def jac(params: np.ndarray) -> np.ndarray:
                        gradients = self.compute_gradients(params)
                        self._last_gradients = gradients
                        self._detect_plateau(gradients)
                        return gradients
                else:
                    jac = None
            result = minimize(
                fun=objective,
                x0=initial_params,
                method=optimizer.value,
                jac=jac,
                tol=tol,
                options={'maxiter': max_iter},
            )
            execution_time = time() - self.start_time
            if self.verbose:
                self._print_summary(result, execution_time)
            return self._build_result(result, execution_time)
        
        def _evaluate_energy(self, params: np.ndarray) -> float:
            self.energy_eval_count += 1
            self.backend.clear_circuit()
            self.backend.reset_state()
            self.ansatz(self.backend, params)
            self.backend.execute_circuit()
            energy = self.backend.compute_expectation(self.hamiltonian)
            return energy

        def compute_gradients(self, params: np.ndarray) -> np.ndarray:
            self.gradient_eval_count += 1
            if self.gradient_method == "parameter_shift":
                return self._parameter_shift_gradients(params)
            elif self.gradient_method == "finite_diff":
                return self._finite_difference_gradients(params)
            else:
                raise ValueError(f"Unknown gradient method: {self.gradient_method}")

        def _parameter_shift_gradients(self, params: np.ndarray, shift: float = np.pi / 2) -> np.ndarray:
            gradients = np.zeros_like(params)
            for i in range(len(params)):
                params_plus = params.copy()
                params_plus[i] += shift
                energy_plus = self._evaluate_energy(params_plus)
                params_minus = params.copy()
                params_minus[i] -= shift
                energy_minus = self._evaluate_energy(params_minus)
                gradients[i] = (energy_plus - energy_minus) / (2 * np.sin(shift))
            return gradients

        def _finite_difference_gradients(self, params: np.ndarray, epsilon: float = 1e-5) -> np.ndarray:
            gradients = np.zeros_like(params)
            for i in range(len(params)):
                params_plus = params.copy()
                params_plus[i] += epsilon
                energy_plus = self._evaluate_energy(params_plus)
                params_minus = params.copy()
                params_minus[i] -= epsilon
                energy_minus = self._evaluate_energy(params_minus)
                gradients[i] = (energy_plus - energy_minus) / (2 * epsilon)
            return gradients

        def _detect_plateau(self, gradients: np.ndarray):
            grad_variance = np.var(gradients)
            grad_norm = np.linalg.norm(gradients)
            self.gradient_variances.append(grad_variance)
            if grad_variance < self.plateau_threshold or grad_norm < self.plateau_threshold:
                self.plateau_iterations.append(self.iteration)
                if self.verbose:
                    print(f"Possible plateau detected (var={grad_variance:.2e}, norm={grad_norm:.2e})")

        def _track_iteration(self, params: np.ndarray, energy: float):
            self.iteration += 1
            self.energy_history.append(energy)
            self.param_history.append(params.copy())

        def _reset_tracking(self):
            self.energy_history = []
            self.param_history = []
            self.gradient_history = []
            self.iteration = 0
            self.energy_eval_count = 0
            self.gradient_eval_count = 0
            self.plateau_iterations = []
            self.gradient_variances = []

        def _print_header(self, initial_params: np.ndarray, optimizer: OptimizerType):
            print("=" * 50)
            print("VQE OPTIMIZATION")
            print("=" * 50)
            print(f"Backend:{self.backend.name}")
            print(f"Hamiltonian:{self.hamiltonian.shape[0]}×{self.hamiltonian.shape[1]}")
            print(f"Parameters:{len(initial_params)}")
            print(f"Optimizer:{optimizer.value}")
            print(f"Gradient:{self.gradient_method}")
            print(f"Plateau check:{'enabled' if self.plateau_threshold else 'disabled'}")
            print("=" * 50)
            print()

        def _print_progress(self):
            if len(self.energy_history) == 0:
                return
            current_energy = self.energy_history[-1]
            print(f"Iter {self.iteration:4d}: E = {current_energy:+.8f}")

        def _print_summary(self, result, execution_time: float):
            print()
            print("=" * 50)
            print("OPTIMIZATION COMPLETE")
            print("=" * 50)
            print(f"Ground state energy: {result.fun:+.8f}")
            print(f"Success: {result.success}")
            print(f"Message: {result.message}")
            print(f"Iterations: {self.iteration}")
            print(f"Energy evaluations: {self.energy_eval_count}")
            print(f"Gradient evaluations: {self.gradient_eval_count}")
            print(f"Execution time: {execution_time:.2f} seconds")
            if self.plateau_iterations:
                print(f"\nPlateau detected at iterations: {self.plateau_iterations}")
            print("=" * 50)

        def _build_result(self, scipy_result, execution_time: float) -> VQEResult:
            return VQEResult(
                optimal_energy=float(scipy_result.fun),
                optimal_params=scipy_result.x,
                energy_history=np.array(self.energy_history),
                param_history=np.array(self.param_history),
                gradient_history=np.array(self.gradient_history) if self.gradient_history else None,
                iterations=self.iteration,
                success=scipy_result.success,
                message=scipy_result.message,
                execution_time=execution_time,
                energy_evaluations=self.energy_eval_count,
                gradient_evaluations=self.gradient_eval_count,
                plateau_detected=len(self.plateau_iterations) > 0,
                plateau_iterations=self.plateau_iterations,
                gradient_variance=np.array(self.gradient_variances) if self.gradient_variances else None,
                backend_name=self.backend.name
            )

        def compute_exact_ground_state(self) -> Tuple[float, np.ndarray]:
            eigenvalues, eigenvectors = np.linalg.eigh(self.hamiltonian)
            ground_energy = eigenvalues[0]
            ground_state = eigenvectors[:, 0]
            return ground_energy, ground_state

        def validate_result(self, result: VQEResult, tolerance: float = 1e-3) -> Dict[str, Any]:
            exact_energy, _ = self.compute_exact_ground_state()
            error = abs(result.optimal_energy - exact_energy)
            relative_error = error / abs(exact_energy) if exact_energy != 0 else error
            converged = error < tolerance
            return {
                'exact_energy': exact_energy,
                'vqe_energy': result.optimal_energy,
                'error': error,
                'relative_error': relative_error,
                'converged': converged,
                'tolerance': tolerance
            }