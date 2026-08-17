import pytest
from backend.domain.models.parameter import Parameter, ParameterVector


class TestParameter:

    def test_basic_parameter_creation(self):
        p = Parameter(name="theta")
        assert p.name == "theta"
        assert p.value == 0.0
        assert p.trainable is True

    def test_create_parameter_with_value(self):
        p = Parameter(name="theta", value=1.5)
        assert p.value == 1.5

    def test_valid_bounds(self):
        p = Parameter(name="theta", value=0.5, lower_bound=0.0, upper_bound=1.0)
        assert p.lower_bound == 0.0
        assert p.upper_bound == 1.0

    def test_lower_bound_greater_than_upper_bound_fails(self):
        with pytest.raises(ValueError):
            Parameter(name="theta", lower_bound=1.0, upper_bound=0.0)

    def test_out_of_bounds_value_fails(self):
        with pytest.raises(ValueError):
            Parameter(name="theta", value=2.0, upper_bound=1.0)

    def test_unique_id_per_parameter(self):
        p1 = Parameter(name="theta")
        p2 = Parameter(name="theta")
        assert p1.id != p2.id


class TestParameterVector:
        def test_create_basic_vector(self):
            v = ParameterVector(name="theta", size=4)
            assert len(v) == 4
            assert v.initial_values == [0.0, 0.0, 0.0, 0.0]

        def test_create_vector_with_values(self):
            v = ParameterVector(name="theta", size=3, initial_values=[0.1, 0.2, 0.3])
            assert v.initial_values == [0.1, 0.2, 0.3]

        def test_inconsistent_values_and_size_fails(self):
            with pytest.raises(ValueError):
                ParameterVector(name="theta", size=4, initial_values=[0.1, 0.2])

        def test_to_parameters(self):
            v = ParameterVector(name="theta", size=3, initial_values=[0.1, 0.2, 0.3])
            params = v.to_parameters()
            assert len(params) == 3
            assert params[0].name == "theta[0]"
            assert params[0].value == 0.1
            assert params[2].name == "theta[2]"