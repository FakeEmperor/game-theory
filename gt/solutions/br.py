"""
Brown robinson solver
"""
from typing import NamedTuple, List, Tuple, Sequence
import numpy as np
import logging
from dataclasses import dataclass
from fractions import Fraction


class Strategies(NamedTuple):
    A: Sequence[float]
    B: Sequence[float]


_logger = logging.getLogger(__name__)


@dataclass
class BrownRobinsonOptimizer:
    strategy_costs: np.ndarray
    max_steps: int = None
    verbose: bool = False

    class OptimizerState(NamedTuple):
        step: int
        game_cost_current: float
        A_income: np.ndarray
        B_loss: np.ndarray
        A_frequencies: np.ndarray
        B_frequencies: np.ndarray
        A_strategy: int
        B_strategy: int
        min_win: float
        max_loss: float

    def __post_init__(self):
        if self.strategy_costs is None:
            raise ValueError("Strategy Costs should not be None!")
        elif len(self.strategy_costs.shape) != 2 or self.strategy_costs.shape[0] != self.strategy_costs.shape[1]:
            raise ValueError(f"This implementation works only for two players, "
                             f"so strategy costs should be a 2D square matrix, "
                             f"got: {self.strategy_costs.shape}.")
        m, n = self.strategy_costs.shape
        self._A_income = np.zeros((1, m)).ravel()
        self._B_loss = np.zeros((1, n)).ravel()
        self._A_strategy_freqs = np.zeros((1, m))
        self._B_strategy_freqs = np.zeros((1, n))
        self._min_win = np.inf
        self._max_loss = -np.inf
        self._A_previous_strategy = 0
        self._B_previous_strategy = 0

        self._steps = 0
        if not self.verbose:
            _logger.setLevel(logging.CRITICAL)
        else:
            _logger.setLevel(logging.DEBUG)

    def step(self) -> float:
        self._steps += 1
        if self._steps == 1:
            a_strategy_most_promising = self._A_previous_strategy
            b_strategy_most_promising = self._B_previous_strategy
        else:
            a_strategy_most_promising = self.A_strategy
            b_strategy_most_promising = self.B_strategy
            self._A_previous_strategy = a_strategy_most_promising
            self._B_previous_strategy = b_strategy_most_promising

        _logger.info(f"[step {self._steps}] Most promising strategy for A: {a_strategy_most_promising}")
        _logger.info(f"[step {self._steps}] Most promising strategy for B: {b_strategy_most_promising}")

        self._A_strategy_freqs[0][a_strategy_most_promising] += 1
        self._B_strategy_freqs[0][b_strategy_most_promising] += 1
        _logger.info(f"[step {self._steps}] Strategy freqs for A: {self._A_strategy_freqs}")
        _logger.info(f"[step {self._steps}] Strategy freqs for B: {self._B_strategy_freqs}")

        self._A_income = (self.strategy_costs @ self._B_strategy_freqs.T).T.ravel()
        self._B_loss = (self._A_strategy_freqs @ self.strategy_costs).ravel()

        _logger.info(f"[step {self._steps}] Income for A: {self._A_income}")
        _logger.info(f"[step {self._steps}] Loss   for B: {self._B_loss}")

        max_win = Fraction(int(np.max(self._A_income)), self._steps)
        min_loss = Fraction(int(np.min(self._B_loss)), self._steps)
        _logger.info(f"[step {self._steps}] Max income for A: {max_win}")
        _logger.info(f"[step {self._steps}] Min loss   for B: {min_loss}")

        self._min_win = min(self._min_win, max_win)
        self._max_loss = max(self._max_loss, min_loss)
        _logger.info(f"[step {self._steps}] Global win  for A: {self._min_win}")
        _logger.info(f"[step {self._steps}] Global loss for B: {self._max_loss}")
        return self.game_cost_error

    def _strategy(self, profits: np.ndarray, select, previous: int, name: str) -> int:
        most_promising = select(profits.ravel())
        all_occurancies = np.argwhere(profits == profits[most_promising]).flatten()
        _logger.info(f"[step {self._steps}] Most promising for {name}: {most_promising}[{most_promising + 1}] "
                     f"(profit: {profits[most_promising]}, all occurancies: {all_occurancies})")
        if len(all_occurancies) > 1:
            _logger.info(f"[step {self._steps}] Returning tie breaker for {name}: "
                         f"{all_occurancies[1]} (next item in the list of candidates)")
            return all_occurancies[1]
        return most_promising

    @property
    def A_strategy(self) -> int:
        return self._strategy(self._A_income, np.argmax, self._A_previous_strategy, "A")

    @property
    def B_strategy(self) -> int:
        return self._strategy(self._B_loss, np.argmin, self._B_previous_strategy, "B")

    @property
    def game_cost_error(self) -> float:
        return self._min_win - self._max_loss

    @property
    def state(self) -> 'OptimizerState':
        return self.OptimizerState(
            step=self._steps,
            game_cost_current=self.game_cost_error,
            A_income=self._A_income,
            B_loss=self._B_loss,
            A_strategy=self._A_previous_strategy,
            B_strategy=self._B_previous_strategy,
            A_frequencies=self._A_strategy_freqs,
            B_frequencies=self._B_strategy_freqs,
            min_win=self._min_win,
            max_loss=self._max_loss,
        )

    def fit(self, game_cost_threshold: float) -> Tuple[Strategies, List[OptimizerState]]:
        game_cost_error_current = np.inf
        game_states = []
        while (game_cost_error_current > game_cost_threshold
               and (self.max_steps is None or self._steps < self.max_steps)):
            game_cost_error_current = self.step()
            game_states.append(self.state)
            _logger.info(f"[step {self._steps}] Game cost err: {game_cost_error_current}")
            _logger.info("===========")
        B = [Fraction(int(f), self._steps) for f in self._B_strategy_freqs.flatten()]
        A = [Fraction(int(f), self._steps) for f in self._A_strategy_freqs.flatten()]
        _logger.info(f"Fictitious play method: A* : {A}, B* : {B}, "
                     f"max loss/profit: {float(self._max_loss):.2f}/{float(self._min_win):.2f}")
        A = np.array(A, dtype=float)
        B = np.array(B, dtype=float)
        _logger.info(f"Fictitious play method: A* : {A}, B* : {B}, "
                     f"max loss/profit: {float(self._max_loss):.2f}/{float(self._min_win):.2f}")
        return (
            Strategies(A=A, B=B),
            game_states
        )
