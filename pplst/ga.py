from .condition import Condition
from .hyperparams import get_hyperparam as get_hp
from .indiv import Indiv
from .rng import get_rng

_MIN_TOURN_SIZE = 2


def tournament_selection(pop):
    tourn_size = get_hp("tourn_size")
    assert tourn_size >= _MIN_TOURN_SIZE

    def _select_random(pop):
        idx = get_rng().randint(0, len(pop))
        return pop[idx]

    best = _select_random(pop)
    for _ in range(_MIN_TOURN_SIZE, (tourn_size + 1)):
        indiv = _select_random(pop)
        if indiv.fitness > best.fitness:
            best = indiv
    return best


def crossover(parent_a, parent_b, selectable_actions):
    if get_rng().random() < get_hp("p_cross"):
        return _uniform_crossover_on_rules(parent_a, parent_b,
                                           selectable_actions)
    else:
        return _clone_parents(parent_a, parent_b, selectable_actions)


def _uniform_crossover_on_rules(parent_a, parent_b, selectable_actions):
    """Uniform crossover with swapping acting on whole rules within indivs."""
    num_rules = get_hp("indiv_size")

    assert len(parent_a.rules) == num_rules
    assert len(parent_b.rules) == num_rules
    child_a_rules = parent_a.rules
    child_b_rules = parent_b.rules

    for idx in range(0, num_rules):
        if get_rng().random() < get_hp("p_cross_swap"):
            _swap(child_a_rules, child_b_rules, idx)
    assert len(child_a_rules) == num_rules
    assert len(child_b_rules) == num_rules

    child_a = Indiv(child_a_rules, selectable_actions)
    child_b = Indiv(child_b_rules, selectable_actions)
    return (child_a, child_b)


def _swap(seq_a, seq_b, idx):
    seq_a[idx], seq_b[idx] = seq_b[idx], seq_a[idx]


def _clone_parents(parent_a, parent_b, selectable_actions):
    """Re-make Indiv objects so ids can be updated properly."""
    child_a = Indiv(parent_a.rules, selectable_actions)
    child_b = Indiv(parent_b.rules, selectable_actions)
    return (child_a, child_b)


def mutate(indiv, encoding):
    """Mutates condition and action of rules contained within indiv by
    resetting them in Rule object."""
    for rule in indiv.rules:
        cond_alleles = rule.condition.alleles
        mut_cond_alleles = encoding.mutate_condition_alleles(cond_alleles)
        mut_cond = Condition(mut_cond_alleles, encoding)
        mut_action = _mutate_action(rule.action, indiv.selectable_actions)
        rule.condition = mut_cond
        rule.action = mut_action


def _mutate_action(action, selectable_actions):
    if get_rng().random() < get_hp("p_mut"):
        other_actions = list(set(selectable_actions) - {action})
        return get_rng().choice(other_actions)
    else:
        return action
