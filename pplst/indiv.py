import abc

from .error import UnsetPropertyError
from .inference import infer_action
from .ids import get_next_indiv_id
from .hyperparams import get_hyperparam as get_hp


def make_indiv(rules, selectable_actions):
    use_policy_cache = get_hp("use_indiv_policy_cache")
    if use_policy_cache:
        cls = PolicyCacheIndiv
    else:
        cls = Indiv
    return cls(rules, selectable_actions)


class IndivABC(metaclass=abc.ABCMeta):
    def __init__(self, rules, selectable_actions):
        self._rules = list(rules)
        self._selectable_actions = selectable_actions
        # *most recent* perf assessment result
        self._perf_assessment_res = None
        self._id = get_next_indiv_id()
        # cache x_nought so inference can be done after pickling without
        # relying on global hp registry
        self._x_nought = get_hp("x_nought")

    @property
    def rules(self):
        return self._rules

    @property
    def selectable_actions(self):
        return self._selectable_actions

    @property
    def perf_assessment_res(self):
        return self._perf_assessment_res

    @perf_assessment_res.setter
    def perf_assessment_res(self, val):
        self._perf_assessment_res = val

    @property
    def fitness(self):
        if self._perf_assessment_res is None:
            raise UnsetPropertyError
        else:
            # fitness == perf
            return self._perf_assessment_res.perf

    @property
    def id(self):
        return self._id

    @property
    def x_nought(self):
        return self._x_nought

    def select_action(self, obs):
        """Performs inference on obs using rules to predict an action;
        i.e. making Indiv act as a policy."""
        raise NotImplementedError

    def __len__(self):
        return len(self._rules)


class Indiv(IndivABC):
    def select_action(self, obs):
        return infer_action(self, obs)


class PolicyCacheIndiv(IndivABC):
    def __init__(self, rules, selectable_actions):
        super().__init__(rules, selectable_actions)
        self._policy_cache = {}

    def select_action(self, obs):
        obs_hashable = tuple(obs)
        try:
            return self._policy_cache[obs_hashable]
        except KeyError:
            action = infer_action(self, obs)
            self._policy_cache[obs_hashable] = action
            return action
