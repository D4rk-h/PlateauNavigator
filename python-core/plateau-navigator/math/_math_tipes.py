from typing import Callable
from model.non_gaussian import NonGaussianOperator


class Operator:
    def __init__(self, n_optical_modes: int, ng_operator: NonGaussianOperator, ordered_product: Callable):
        self.n_optical_modes = n_optical_modes # order matters, create a well done method that lists em from optical modes
        self.ng_operator = ng_operator
        self.ordered_product = ordered_product

    def _list_optical_modes(self) -> list[int]:
        return self._make_split_list(self.n_optical_modes)
    
    def _make_split_list(total_number):
        half = total_number // 2
        return list(range(1, half + 1)) * 2


if __name__ == "__main__":
    pass