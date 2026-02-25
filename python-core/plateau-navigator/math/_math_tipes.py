class Operator():
    def __init__(self, n_optical_modes: int, name: str):
        self.n_optical_modes = n_optical_modes
        self.name = name

    def _make_split_list(self, total_number):
        half = total_number // 2
        return list(range(1, half + 1)) * 2

    def _adjoint(self):
        pass

    def _is_hermitian(self):
        pass

    def _get_n_modes():
        return self.n_optical_modes
    
     


if __name__ == "__main__":
    op = Operator(6, "a")
    print(op._list_optical_modes())
