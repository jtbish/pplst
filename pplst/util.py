import numpy as np


def augment_obs(obs, x_nought):
    return np.concatenate(([x_nought], obs))
