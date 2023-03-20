import numpy as np

from .hyperparams import get_hyperparam as get_hp
from .rng import get_rng

np.seterr(divide="raise", over="raise", invalid="raise")

_INIT_PAYOFF_VAR = 0
_INIT_PAYOFF_STDEV = 0


class Rule:
    def __init__(self, condition, action):
        self._condition = condition
        self._action = action

        self._num_features = len(condition)
        self._weight_vec = self._init_weight_vec(self._num_features)
        self._payoff_var = _INIT_PAYOFF_VAR
        self._payoff_stdev = _INIT_PAYOFF_STDEV

    def _init_weight_vec(self, num_features):
        # since linear prediction only,
        # weight vec is of len n+1, n = num features
        low = get_hp("weight_I_min")
        high = get_hp("weight_I_max")
        assert low <= high
        return get_rng().uniform(low, high,
                                 size=(num_features + 1)).astype(np.float32)

    @property
    def condition(self):
        return self._condition

    @condition.setter
    def condition(self, val):
        self._condition = val

    @property
    def action(self):
        return self._action

    @action.setter
    def action(self, val):
        self._action = val

    @property
    def weight_vec(self):
        return self._weight_vec

    @weight_vec.setter
    def weight_vec(self, val):
        self._weight_vec = val

    @property
    def payoff_var(self):
        return self._payoff_var

    @payoff_var.setter
    def payoff_var(self, val):
        self._payoff_var = val

    @property
    def payoff_stdev(self):
        return self._payoff_stdev

    @payoff_stdev.setter
    def payoff_stdev(self, val):
        self._payoff_stdev = val

    def does_match(self, obs):
        return self._condition.does_match(obs)

    def prediction(self, aug_obs):
        return np.dot(aug_obs, self._weight_vec)

    def strength(self, aug_obs):
        """Strength is computed based on given obs"""
        return self.prediction(aug_obs) - self._payoff_stdev

    def __eq__(self, other):
        return (self._condition == other._condition and
                self._action == other._action and
                np.array_equal(self._weight_vec, other._weight_vec) and
                self._payoff_var == other._payoff_var and
                self._payoff_stdev == other._payoff_stdev)

    def __str__(self):
        return f"{self._condition} -> {self._action}"
