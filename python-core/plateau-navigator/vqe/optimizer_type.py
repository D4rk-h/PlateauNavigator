from enum import Enum


class OptimizerType(Enum):
    COBYLA = "COBYLA"
    SLSQP = "SLSQP"
    BFGS = "BFGS"
    L_BFGS_B = "L-BFGS-B"
    NELDER_MEAD = "Nelder-Mead"
    POWELL = "Powell"