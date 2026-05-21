"""Тесты множителей M(S)."""
import math

import pytest

from engine.multiplier import (
    CappedMultiplier,
    ConstantMultiplier,
    LinearMultiplier,
    LogMultiplier,
    PowerMultiplier,
)


class TestConstantMultiplier:
    def test_default_is_one(self):
        m = ConstantMultiplier()
        assert m(0) == 1.0
        assert m(1000) == 1.0

    def test_custom_value(self):
        m = ConstantMultiplier(c=2.5)
        assert m(0) == pytest.approx(2.5)
        assert m(999) == pytest.approx(2.5)

    def test_repr_contains_value(self):
        assert "2.5" in ConstantMultiplier(c=2.5).name


class TestLinearMultiplier:
    def test_at_zero(self):
        m = LinearMultiplier(k=100)
        assert m(0) == pytest.approx(1.0)

    def test_at_k(self):
        m = LinearMultiplier(k=100)
        assert m(100) == pytest.approx(2.0)

    def test_monotone_increasing(self):
        m = LinearMultiplier(k=50)
        values = [m(s) for s in range(0, 500, 50)]
        assert all(values[i] < values[i + 1] for i in range(len(values) - 1))

    def test_invalid_k(self):
        with pytest.raises(ValueError):
            LinearMultiplier(k=0)
        with pytest.raises(ValueError):
            LinearMultiplier(k=-1)


class TestPowerMultiplier:
    def test_at_zero(self):
        m = PowerMultiplier(k=100, alpha=0.5)
        assert m(0) == pytest.approx(1.0)

    def test_alpha_one_matches_linear(self):
        lin = LinearMultiplier(k=100)
        pw = PowerMultiplier(k=100, alpha=1.0)
        for s in [0, 50, 100, 200]:
            assert pw(s) == pytest.approx(lin(s))

    def test_sublinear_less_than_linear(self):
        lin = LinearMultiplier(k=100)
        pw = PowerMultiplier(k=100, alpha=0.5)
        for s in [100, 200, 500]:
            assert pw(s) < lin(s)

    def test_superlinear_greater_than_linear(self):
        lin = LinearMultiplier(k=100)
        pw = PowerMultiplier(k=100, alpha=2.0)
        for s in [100, 200, 500]:
            assert pw(s) > lin(s)

    def test_monotone_increasing(self):
        m = PowerMultiplier(k=100, alpha=0.5)
        values = [m(s) for s in range(0, 600, 100)]
        assert all(values[i] < values[i + 1] for i in range(len(values) - 1))


class TestLogMultiplier:
    def test_at_zero(self):
        m = LogMultiplier(k=100)
        assert m(0) == pytest.approx(1.0)

    def test_slower_than_linear(self):
        lin = LinearMultiplier(k=100)
        log = LogMultiplier(k=100)
        for s in [200, 500, 1000]:
            assert log(s) < lin(s)

    def test_monotone_increasing(self):
        m = LogMultiplier(k=100)
        values = [m(s) for s in range(0, 600, 100)]
        assert all(values[i] < values[i + 1] for i in range(len(values) - 1))

    def test_formula(self):
        m = LogMultiplier(k=100)
        s = 300
        expected = 1.0 + math.log1p(300 / 100)
        assert m(s) == pytest.approx(expected)


class TestCappedMultiplier:
    def test_cap_enforced(self):
        inner = LinearMultiplier(k=10)
        capped = CappedMultiplier(inner, cap=5.0)
        assert capped(1000) == pytest.approx(5.0)

    def test_below_cap_equals_inner(self):
        inner = LinearMultiplier(k=100)
        capped = CappedMultiplier(inner, cap=50.0)
        for s in [0, 10, 50]:
            assert capped(s) == pytest.approx(inner(s))

    def test_at_zero(self):
        capped = CappedMultiplier(LinearMultiplier(k=100), cap=50.0)
        assert capped(0) == pytest.approx(1.0)

    def test_repr_contains_cap(self):
        capped = CappedMultiplier(LinearMultiplier(k=100), cap=50.0)
        assert "50" in capped.name
