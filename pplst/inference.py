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
        (action_sets, reprd_actions) = _gen_action_sets(
            match_set, selectable_actions=indiv.selectable_actions)

        num_reprd_actions = len(reprd_actions)
        assert num_reprd_actions > 0
        is_action_conflict = (num_reprd_actions > 1)

        if is_action_conflict:
            # use strength to resolve conflict
            aug_obs = augment_obs(obs, x_nought=indiv.x_nought)
            best_action = _get_best_action(action_sets, reprd_actions, aug_obs)
            action_set = action_sets[best_action]
        else:
            assert num_reprd_actions == 1
            # use sole action represented
            best_action = list(reprd_actions)[0]
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
    reprd_actions = set()
    for rule in match_set:
        a = rule.action
        action_sets[a].append(rule)
        reprd_actions.add(a)
    return (action_sets, reprd_actions)


def _get_best_action(action_sets, reprd_actions, aug_obs):
    # pick action with highest strength via double max: first over rule
    # strengths for each action, then over max strengths of all actions
    max_a_strengths = _get_max_action_strengths(action_sets, reprd_actions,
                                                aug_obs)
    return max(max_a_strengths, key=max_a_strengths.get)


def _get_max_action_strengths(action_sets, reprd_actions, aug_obs):
    max_a_strengths = OrderedDict()
    for a in reprd_actions:
        action_set = action_sets[a]
        max_a_strengths[a] = max(
            [rule.strength(aug_obs) for rule in action_set])
    return max_a_strengths


def _is_empty(set_):
    return len(set_) == 0
