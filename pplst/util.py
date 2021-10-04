import numpy as np

from .hyperparams import get_hyperparam as get_hp


def augment_obs(obs):
    return np.concatenate(([get_hp("x_nought")], obs))
