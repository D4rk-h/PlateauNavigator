from abc import ABC


class NonGaussianOperator(ABC):
    def __init__(self, name: str):
        self.name = name