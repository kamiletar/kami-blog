"""Тесты стратегий: минимум 1 матч против AllC и AllD, проверка первых N ходов."""
import numpy as np
import pytest

from engine.action import Action
from engine.game import Game
from engine.multiplier import ConstantMultiplier
from strategies.basic import AllC, AllD, Random
from strategies.reactive import (
    ContriteTfT,
    GenerousTfT,
    GrimTrigger,
    Pavlov,
    SuspiciousTfT,
    TitForTat,
    TitForTwoTats,
)

M1 = ConstantMultiplier(1.0)


def play(strat_a, strat_b, rounds=20, seed=0):
    return Game(strat_a, strat_b, rounds=rounds, multiplier=M1, seed=seed).play()


# ─── AllC / AllD ────────────────────────────────────────────────────────────

class TestAllC:
    def test_vs_allc(self):
        r = play(AllC(), AllC())
        assert np.all(r.history_a == Action.COOPERATE)

    def test_vs_alld(self):
        r = play(AllC(), AllD())
        assert np.all(r.history_a == Action.COOPERATE)
        assert r.total_a == 0.0  # всегда sucker

    def test_coop_rate_is_one(self):
        assert play(AllC(), AllD()).coop_rate_a == 1.0


class TestAllD:
    def test_vs_allc(self):
        r = play(AllD(), AllC())
        assert np.all(r.history_a == Action.DEFECT)
        assert r.total_a == pytest.approx(100.0)  # 20 * 5

    def test_vs_alld(self):
        r = play(AllD(), AllD())
        assert np.all(r.history_a == Action.DEFECT)
        assert r.total_a == pytest.approx(20.0)  # 20 * 1


# ─── Random ─────────────────────────────────────────────────────────────────

class TestRandom:
    def test_reproducible(self):
        r1 = play(Random(seed=42), AllC())
        r2 = play(Random(seed=42), AllC())
        np.testing.assert_array_equal(r1.history_a, r2.history_a)

    def test_different_seeds_differ(self):
        r1 = play(Random(seed=1), AllC(), rounds=50)
        r2 = play(Random(seed=2), AllC(), rounds=50)
        assert not np.array_equal(r1.history_a, r2.history_a)

    def test_p0_always_defects(self):
        r = play(Random(p=0.0, seed=0), AllC(), rounds=30)
        assert np.all(r.history_a == Action.DEFECT)

    def test_p1_always_cooperates(self):
        r = play(Random(p=1.0, seed=0), AllC(), rounds=30)
        assert np.all(r.history_a == Action.COOPERATE)

    def test_invalid_p(self):
        with pytest.raises(ValueError):
            Random(p=1.5)


# ─── TitForTat ───────────────────────────────────────────────────────────────

class TestTitForTat:
    def test_first_move_is_c(self):
        r = play(TitForTat(), AllD())
        assert r.history_a[0] == Action.COOPERATE

    def test_vs_alld_mirrors_after_first(self):
        r = play(TitForTat(), AllD(), rounds=10)
        # Раунд 0: C; раунды 1-9: D (mirror AllD)
        assert r.history_a[0] == Action.COOPERATE
        assert np.all(r.history_a[1:] == Action.DEFECT)

    def test_vs_allc_always_cooperates(self):
        r = play(TitForTat(), AllC(), rounds=10)
        assert np.all(r.history_a == Action.COOPERATE)

    def test_reset_clears_no_state(self):
        tft = TitForTat()
        play(tft, AllD())
        tft.reset()
        r2 = play(tft, AllC())
        assert r2.history_a[0] == Action.COOPERATE


# ─── SuspiciousTfT ───────────────────────────────────────────────────────────

class TestSuspiciousTfT:
    def test_first_move_is_d(self):
        r = play(SuspiciousTfT(), AllC())
        assert r.history_a[0] == Action.DEFECT

    def test_vs_allc_mirrors_then_cooperates(self):
        r = play(SuspiciousTfT(), AllC(), rounds=5)
        assert r.history_a[0] == Action.DEFECT
        # Раунды 1-4: C (mirror AllC)
        assert np.all(r.history_a[1:] == Action.COOPERATE)


# ─── TitForTwoTats ───────────────────────────────────────────────────────────

class TestTitForTwoTats:
    def test_tolerates_single_defect(self):
        """Одна дефекция соперника не вызывает ответа."""
        from strategies.basic import AllC

        # Подготовим специальный ответчик: C на 0-9, D на 10-й, C дальше
        class OneDefect(AllC):
            def play(self, my, opp, t, T):
                return Action.DEFECT if t == 5 else Action.COOPERATE

        r = play(TitForTwoTats(), OneDefect(), rounds=15)
        # TFT2 никогда не должен дефектировать — одна D не триггер
        assert np.all(r.history_a == Action.COOPERATE)

    def test_responds_to_two_consecutive(self):
        """Две подряд D соперника вызывают ответную D."""

        class TwoConsecutive(AllC):
            def play(self, my, opp, t, T):
                return Action.DEFECT if t >= 3 else Action.COOPERATE

        r = play(TitForTwoTats(), TwoConsecutive(), rounds=10)
        # С раунда 5 (два D подряд наблюдено) TFT2 должен дефектировать
        assert r.history_a[5] == Action.DEFECT


# ─── GrimTrigger ─────────────────────────────────────────────────────────────

class TestGrimTrigger:
    def test_vs_allc_always_cooperates(self):
        r = play(GrimTrigger(), AllC(), rounds=20)
        assert np.all(r.history_a == Action.COOPERATE)

    def test_triggered_by_first_defect(self):
        r = play(GrimTrigger(), AllD(), rounds=10)
        # Раунд 0: C (ещё не триггернулся)
        assert r.history_a[0] == Action.COOPERATE
        # Раунды 1-9: D навсегда
        assert np.all(r.history_a[1:] == Action.DEFECT)

    def test_reset_clears_trigger(self):
        grim = GrimTrigger()
        play(grim, AllD())
        grim.reset()
        r2 = play(grim, AllC(), rounds=5)
        assert np.all(r2.history_a == Action.COOPERATE)


# ─── GenerousTfT ─────────────────────────────────────────────────────────────

class TestGenerousTfT:
    def test_vs_allc_always_cooperates(self):
        r = play(GenerousTfT(seed=0), AllC(), rounds=20)
        assert np.all(r.history_a == Action.COOPERATE)

    def test_forgiveness_zero_equals_tft(self):
        """forgiveness=0 → поведение идентично TfT."""
        gtft = GenerousTfT(forgiveness=0.0, seed=0)
        tft = TitForTat()
        r_gtft = play(gtft, AllD(), rounds=15)
        r_tft = play(tft, AllD(), rounds=15)
        np.testing.assert_array_equal(r_gtft.history_a, r_tft.history_a)

    def test_invalid_forgiveness(self):
        with pytest.raises(ValueError):
            GenerousTfT(forgiveness=-0.1)


# ─── Pavlov ──────────────────────────────────────────────────────────────────

class TestPavlov:
    def test_first_move_is_c(self):
        r = play(Pavlov(), AllD())
        assert r.history_a[0] == Action.COOPERATE

    def test_vs_allc_always_cooperates(self):
        """CC → stay → C бесконечно."""
        r = play(Pavlov(), AllC(), rounds=20)
        assert np.all(r.history_a == Action.COOPERATE)

    def test_vs_alld_alternates(self):
        """Pavlov vs AllD: C,D,C,D,...

        Раунд 0: намерение C (первый ход).
        Раунд 1: оппонент сыграл D → shift → D.
        Раунд 2: оппонент сыграл D → shift → C.
        Раунд 3: оппонент сыграл D → shift → D. И так далее.
        """
        r = play(Pavlov(), AllD(), rounds=8)
        expected = [Action.COOPERATE, Action.DEFECT] * 4
        np.testing.assert_array_equal(r.history_a, expected)


# ─── ContriteTfT ─────────────────────────────────────────────────────────────

class TestContriteTfT:
    def test_vs_allc_always_cooperates(self):
        r = play(ContriteTfT(), AllC(), rounds=20)
        assert np.all(r.history_a == Action.COOPERATE)

    def test_vs_alld_mirrors_like_tft_without_noise(self):
        """Без шума ошибок нет → поведение = TfT."""
        r_ctft = play(ContriteTfT(), AllD(), rounds=10)
        r_tft = play(TitForTat(), AllD(), rounds=10)
        np.testing.assert_array_equal(r_ctft.history_a, r_tft.history_a)

    def test_reset_clears_intentions(self):
        c = ContriteTfT()
        play(c, AllD(), rounds=5)
        c.reset()
        r2 = play(c, AllC(), rounds=5)
        assert r2.history_a[0] == Action.COOPERATE

    def test_contrite_forgives_noise_mistake(self):
        """При noise=1 implementation AllC инвертируется в D.
        ContriteTfT должен распознать ошибку и кооперировать на следующем ходу."""
        g = Game(
            ContriteTfT(), AllC(),
            rounds=4,
            multiplier=M1,
            noise=1.0,
            noise_mode="implementation",
            seed=0,
        )
        r = g.play()
        # При noise=1 все ходы инвертируются: ContriteTfT намеревался C → сыграл D.
        # В следующем раунде должен кооперировать (раскаяние), а не дефектировать.
        # history_a[0] = D (инверсия C), намерение = C → ошибка → intent[1] = C → played D (инверсия)
        # Проверяем что намерение на раунд 1 — C (раскаяние)
        # Нет прямого доступа, но в mode implementation my_view = played.
        # Косвенно: ContriteTfT vs AllC (AllC всегда C, но инвертируется в D).
        # При noise=1 perceived_by_a[t] = history_b[t] = D (инверсия C AllC).
        # Раунд 0: намерение C, played D, perceived_opp = D.
        # Раунд 1: was_mistake=True (intended C, played D) → intent = C, played = D.
        # Намерение постоянно C из-за раскаяния при constant noise=1.
        assert np.all(r.history_a == Action.DEFECT)  # все инвертированы
