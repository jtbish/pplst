import numpy as np

from .hyperparams import get_hyperparam as get_hp
from .util import augment_obs

np.seterr(all="raise")


def update_action_set(action_set, payoff, obs):
    aug_obs = augment_obs(obs)
    proc_obs = _process_aug_obs(aug_obs)
    for rule in action_set:
        _update_payoff_prediction(rule, payoff, aug_obs, proc_obs)
        _update_payoff_var_and_stdev(rule, payoff, aug_obs)


def _process_aug_obs(aug_obs):
    return np.sum(np.square(aug_obs))


def _update_payoff_prediction(rule, payoff, aug_obs, proc_obs):
    """Normalised least mean squares."""
    norm = proc_obs
    error = payoff - rule.prediction(aug_obs)
    correction = (get_hp("eta") / norm) * error
    rule.weight_vec += (aug_obs * correction)


def _update_payoff_var_and_stdev(rule, payoff, aug_obs):
    """As per SAMUEL"""
    eta = get_hp("eta")
    pred = rule.prediction(aug_obs)
    # v_i = (1 - c)*v_i + c*(mu_i - r)^2
    rule.payoff_var = (1 - eta) * rule.payoff_var + eta * (pred - payoff)**2
    rule.payoff_stdev = np.sqrt(rule.payoff_var)
