from collections import OrderedDict

from .util import augment_obs

NULL_ACTION = -1


def infer_action(indiv, obs):
    (action, _) = _infer_action_and_action_set(indiv, obs)
    return action


def infer_action_and_action_set(indiv, obs):
    return _infer_action_and_action_set(indiv, obs)


def _infer_action_and_action_set(indiv, obs):
    match_set = _gen_match_set(indiv, obs)
    if not _is_empty(match_set):
        action_sets = _gen_action_sets(
            match_set, selectable_actions=indiv.selectable_actions)
        aug_obs = augment_obs(obs, x_nought=indiv.x_nought)

        best_action = _get_best_action(action_sets, aug_obs)
        action_set = action_sets[best_action]
    else:
        best_action = NULL_ACTION
        action_set = None
    return (best_action, action_set)


def _gen_match_set(indiv, obs):
    return [rule for rule in indiv.rules if rule.does_match(obs)]


def _gen_action_sets(match_set, selectable_actions):
    """Build up action sets in more intelligent way that only requires
    iterating over match set once."""
    action_sets = OrderedDict({a: [] for a in selectable_actions})
    for rule in match_set:
        a = rule.action
        action_sets[a].append(rule)
    return action_sets


def _get_best_action(action_sets, aug_obs):
    # pick action with highest strength via double max: first over rule
    # strengths for each action, then over max strengths of all actions
    max_a_strengths = _get_max_action_strengths(action_sets, aug_obs)
    return max(max_a_strengths, key=max_a_strengths.get)


def _get_max_action_strengths(action_sets, aug_obs):
    max_a_strengths = OrderedDict()
    for (a, action_set) in action_sets.items():
        if not _is_empty(action_set):
            max_a_strengths[a] = max(
                [rule.strength(aug_obs) for rule in action_set])
    return max_a_strengths


def _is_empty(set_):
    return len(set_) == 0
