"""Тесты игрового движка: матрица, обновление S, шум, воспроизводимость."""
import numpy as np
import pytest

from engine.action import Action
from engine.game import DEFAULT_PAYOFF, Game, GameResult
from engine.multiplier import ConstantMultiplier, LinearMultiplier
from strategies.basic import AllC, AllD


class TestDefaultPayoff:
    """Проверяем корректность матрицы выигрышей по умолчанию."""

    def test_cc(self):
        assert DEFAULT_PAYOFF[0, 0, 0] == 3.0
        assert DEFAULT_PAYOFF[0, 0, 1] == 3.0

    def test_cd(self):
        """A кооперирует, B дефектирует."""
        assert DEFAULT_PAYOFF[0, 1, 0] == 0.0  # A получает 0
        assert DEFAULT_PAYOFF[0, 1, 1] == 5.0  # B получает 5

    def test_dc(self):
        """A дефектирует, B кооперирует."""
        assert DEFAULT_PAYOFF[1, 0, 0] == 5.0
        assert DEFAULT_PAYOFF[1, 0, 1] == 0.0

    def test_dd(self):
        assert DEFAULT_PAYOFF[1, 1, 0] == 1.0
        assert DEFAULT_PAYOFF[1, 1, 1] == 1.0


class TestGameConstantAllC:
    """AllC vs AllC без шума — детерминированный случай."""

    @pytest.fixture
    def result(self):
        g = Game(AllC(), AllC(), rounds=10, multiplier=ConstantMultiplier(1.0), seed=0)
        return g.play()

    def test_all_cooperate(self, result):
        assert np.all(result.history_a == Action.COOPERATE)
        assert np.all(result.history_b == Action.COOPERATE)

    def test_raw_payoff_each_round(self, result):
        assert np.all(result.raw_payoffs_a == 3.0)
        assert np.all(result.raw_payoffs_b == 3.0)

    def test_multiplied_equals_raw_with_constant_one(self, result):
        np.testing.assert_allclose(result.multiplied_payoffs_a, result.raw_payoffs_a)
        np.testing.assert_allclose(result.multiplied_payoffs_b, result.raw_payoffs_b)

    def test_total(self, result):
        assert result.total_a == pytest.approx(30.0)
        assert result.total_b == pytest.approx(30.0)

    def test_coop_rate(self, result):
        assert result.coop_rate_a == 1.0
        assert result.coop_rate_b == 1.0


class TestGameConstantAllD:
    """AllD vs AllD — проверка DD-выигрышей."""

    @pytest.fixture
    def result(self):
        g = Game(AllD(), AllD(), rounds=5, multiplier=ConstantMultiplier(1.0), seed=0)
        return g.play()

    def test_raw_payoff(self, result):
        assert np.all(result.raw_payoffs_a == 1.0)
        assert np.all(result.raw_payoffs_b == 1.0)

    def test_total(self, result):
        assert result.total_a == pytest.approx(5.0)


class TestSTrajectory:
    """Проверяем корректность обновления накопленного счёта S."""

    def test_s_starts_at_zero(self):
        g = Game(AllC(), AllC(), rounds=5, multiplier=ConstantMultiplier(1.0), seed=0)
        r = g.play()
        assert r.S_trajectory[0] == 0.0

    def test_s_update_manual(self):
        """Вручную проверяем первые несколько шагов при M(S)=1."""
        g = Game(AllC(), AllC(), rounds=3, multiplier=ConstantMultiplier(1.0), seed=0)
        r = g.play()
        # Раунд 0: S=0, M=1, pay=3,3 → S обновляется до 6
        assert r.S_trajectory[1] == pytest.approx(6.0)
        # Раунд 1: S=6, M=1, pay=3,3 → S обновляется до 12
        assert r.S_trajectory[2] == pytest.approx(12.0)
        # Раунд 2: S=12, M=1, pay=3,3 → S обновляется до 18
        assert r.S_trajectory[3] == pytest.approx(18.0)

    def test_s_update_with_linear_multiplier(self):
        """S растёт быстрее при LinearMultiplier из-за компаундинга."""
        g_const = Game(AllC(), AllC(), rounds=5, multiplier=ConstantMultiplier(1.0), seed=0)
        g_lin = Game(AllC(), AllC(), rounds=5, multiplier=LinearMultiplier(k=100), seed=0)
        r_c = g_const.play()
        r_l = g_lin.play()
        # При Linear, M > 1 начиная с раунда 1, значит S_lin > S_const для t >= 2
        assert r_l.S_trajectory[-1] > r_c.S_trajectory[-1]

    def test_s_trajectory_length(self):
        g = Game(AllC(), AllC(), rounds=7, multiplier=ConstantMultiplier(1.0), seed=0)
        r = g.play()
        assert len(r.S_trajectory) == 8

    def test_m_trajectory_matches_s(self):
        """M_trajectory[t] должен совпадать с multiplier(S_trajectory[t])."""
        m = LinearMultiplier(k=100)
        g = Game(AllC(), AllC(), rounds=10, multiplier=m, seed=0)
        r = g.play()
        for t in range(10):
            assert r.M_trajectory[t] == pytest.approx(m(r.S_trajectory[t]))


class TestMixedStrategies:
    """AllC vs AllD — проверяем корректность CD/DC выигрышей."""

    def test_allc_vs_alld_payoffs(self):
        g = Game(AllC(), AllD(), rounds=5, multiplier=ConstantMultiplier(1.0), seed=0)
        r = g.play()
        # A=C, B=D: A получает 0, B получает 5
        assert np.all(r.raw_payoffs_a == 0.0)
        assert np.all(r.raw_payoffs_b == 5.0)

    def test_alld_vs_allc_payoffs(self):
        g = Game(AllD(), AllC(), rounds=5, multiplier=ConstantMultiplier(1.0), seed=0)
        r = g.play()
        # A=D, B=C: A получает 5, B получает 0
        assert np.all(r.raw_payoffs_a == 5.0)
        assert np.all(r.raw_payoffs_b == 0.0)


class TestEdgeCases:
    def test_zero_rounds(self):
        g = Game(AllC(), AllC(), rounds=0, multiplier=ConstantMultiplier(1.0), seed=0)
        r = g.play()
        assert len(r.history_a) == 0
        assert len(r.S_trajectory) == 1
        assert r.S_trajectory[0] == 0.0
        assert r.total_a == 0.0

    def test_one_round(self):
        g = Game(AllC(), AllC(), rounds=1, multiplier=ConstantMultiplier(1.0), seed=0)
        r = g.play()
        assert len(r.history_a) == 1
        assert r.raw_payoffs_a[0] == 3.0

    def test_invalid_rounds(self):
        with pytest.raises(ValueError):
            Game(AllC(), AllC(), rounds=-1, multiplier=ConstantMultiplier())

    def test_invalid_noise(self):
        with pytest.raises(ValueError):
            Game(AllC(), AllC(), rounds=5, multiplier=ConstantMultiplier(), noise=1.5)


class TestNoise:
    """Тесты шумового режима."""

    def test_no_noise_no_flip(self):
        """Без шума история совпадает с детерминированной стратегией."""
        g = Game(AllC(), AllC(), rounds=10, multiplier=ConstantMultiplier(), noise=0.0, seed=42)
        r = g.play()
        assert np.all(r.history_a == Action.COOPERATE)

    def test_noise_1_always_flips(self):
        """При noise=1 AllC всегда играет D в implementation режиме."""
        g = Game(
            AllC(), AllC(), rounds=20,
            multiplier=ConstantMultiplier(),
            noise=1.0, noise_mode="implementation", seed=0,
        )
        r = g.play()
        assert np.all(r.history_a == Action.DEFECT)
        assert np.all(r.history_b == Action.DEFECT)

    def test_implementation_both_see_flipped(self):
        """В implementation режиме оба видят одно и то же (инвертированное) действие."""
        g = Game(
            AllC(), AllC(), rounds=50,
            multiplier=ConstantMultiplier(),
            noise=1.0, noise_mode="implementation", seed=7,
        )
        r = g.play()
        # Что A наблюдал у B = что B реально сыграл
        np.testing.assert_array_equal(r.perceived_by_a, r.history_b)
        # Что B наблюдал у A = что A реально сыграл
        np.testing.assert_array_equal(r.perceived_by_b, r.history_a)

    def test_perception_played_equals_intended(self):
        """В perception режиме фактически сыгранное = намерение AllC = всегда C."""
        g = Game(
            AllC(), AllC(), rounds=20,
            multiplier=ConstantMultiplier(),
            noise=0.5, noise_mode="perception", seed=99,
        )
        r = g.play()
        # AllC всегда намеревается C; в perception режиме сыгранное = намерение
        assert np.all(r.history_a == Action.COOPERATE)
        assert np.all(r.history_b == Action.COOPERATE)

    def test_reproducibility_with_seed(self):
        """Одинаковый seed → одинаковый результат."""
        kwargs = dict(rounds=50, multiplier=ConstantMultiplier(), noise=0.1, seed=123)
        r1 = Game(AllC(), AllD(), **kwargs).play()
        r2 = Game(AllC(), AllD(), **kwargs).play()
        np.testing.assert_array_equal(r1.history_a, r2.history_a)
        np.testing.assert_array_equal(r1.history_b, r2.history_b)

    def test_different_seeds_may_differ(self):
        """Разные seed → (почти наверняка) разные результаты при noise > 0."""
        r1 = Game(AllC(), AllC(), rounds=50, multiplier=ConstantMultiplier(),
                  noise=0.5, seed=1).play()
        r2 = Game(AllC(), AllC(), rounds=50, multiplier=ConstantMultiplier(),
                  noise=0.5, seed=2).play()
        assert not np.array_equal(r1.history_a, r2.history_a)


class TestLinearCompounding:
    """Проверяем что компаундинг работает корректно вручную на маленьком примере."""

    def test_manual_three_rounds(self):
        """AllC vs AllC, Linear(k=6), 3 раунда. Считаем вручную.

        Раунд 0: S=0, M=1+0/6=1, raw=3,3, mul=3,3, S_new=6
        Раунд 1: S=6, M=1+6/6=2, raw=3,3, mul=6,6, S_new=6+12=18
        Раунд 2: S=18, M=1+18/6=4, raw=3,3, mul=12,12, S_new=18+24=42
        """
        g = Game(AllC(), AllC(), rounds=3, multiplier=LinearMultiplier(k=6), seed=0)
        r = g.play()
        assert r.S_trajectory[0] == pytest.approx(0.0)
        assert r.S_trajectory[1] == pytest.approx(6.0)
        assert r.S_trajectory[2] == pytest.approx(18.0)
        assert r.S_trajectory[3] == pytest.approx(42.0)
        assert r.M_trajectory[0] == pytest.approx(1.0)
        assert r.M_trajectory[1] == pytest.approx(2.0)
        assert r.M_trajectory[2] == pytest.approx(4.0)
        assert r.multiplied_payoffs_a[0] == pytest.approx(3.0)
        assert r.multiplied_payoffs_a[1] == pytest.approx(6.0)
        assert r.multiplied_payoffs_a[2] == pytest.approx(12.0)
