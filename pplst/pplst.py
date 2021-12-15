import copy
import logging
import os
from collections import namedtuple
from multiprocessing import Pool

from rlenvs.environment import assess_perf

from .ga import crossover, mutate, tournament_selection
from .hyperparams import get_hyperparam as get_hp
from .hyperparams import register_hyperparams
from .inference import NULL_ACTION, infer_action_and_action_set
from .init import init_pop
from .param_update import update_action_set
from .rng import seed_rng

_NUM_CPUS = int(os.environ['SLURM_JOB_CPUS_PER_NODE'])

TrajectoryStep = namedtuple("TrajectoryStep",
                            ["obs", "action", "action_set", "reward"])


class PPLST:
    def __init__(self, reinf_env, perf_env, encoding, hyperparams_dict):
        # environment for doing inner loop: trajectory reinforcements
        self._reinf_env = reinf_env
        # environment for doing perf assessment for GA fitness
        self._perf_env = perf_env
        assert (self._reinf_env.action_space == self._perf_env.action_space)
        self._selectable_actions = self._reinf_env.action_space
        self._encoding = encoding
        self._hyperparams_dict = hyperparams_dict
        register_hyperparams(self._hyperparams_dict)
        seed_rng(get_hp("seed"))
        self._pop = None

    @property
    def pop(self):
        return self._pop

    def init(self):
        self._pop = init_pop(self._encoding, self._selectable_actions)
        self._pop = self._run_pop_learning_parallel(self._pop)
        return self._pop

    def run_gen(self):
        pop_size = get_hp("pop_size")
        assert (pop_size % 2) == 0
        num_breeding_rounds = (pop_size // 2)
        new_pop = []
        for _ in range(num_breeding_rounds):
            parent_a = copy.deepcopy(tournament_selection(self._pop))
            parent_b = copy.deepcopy(tournament_selection(self._pop))
            (child_a, child_b) = crossover(parent_a, parent_b,
                                           self._selectable_actions)
            # check children inited properly after crossover as new objs.
            assert child_a.perf_assessment_res is None
            assert child_b.perf_assessment_res is None

            for child in (child_a, child_b):
                mutate(child, self._encoding)
                new_pop.append(child)

        assert len(new_pop) == pop_size
        self._pop = self._run_pop_learning_parallel(new_pop)
        return self._pop

    def _run_pop_learning_serial(self, pop):
        """For debugging / profiling"""
        updated_pop = [
            self._run_indiv_learning(indiv, self._hyperparams_dict)
            for indiv in pop
        ]
        return updated_pop

    def _run_pop_learning_parallel(self, pop):
        # process parallelism for doing "learning" for each indiv in pop
        with Pool(_NUM_CPUS) as pool:
            updated_pop = pool.starmap(self._run_indiv_learning,
                                       [(indiv, self._hyperparams_dict)
                                        for indiv in pop])

        return updated_pop

    def _run_indiv_learning(self, indiv, hyperparams_dict):
        """'Learning' has two stages: first, update payoff estimates (do MC RL)
        for rules within an Indiv via trajectories in an inner loop.
        Then eval the perf (fitness) of the Indiv as a whole for GA to use."""

        # but first re-register hyperparams globally for this process. (bit
        # hacky, maybe change later)
        register_hyperparams(hyperparams_dict)
        num_reinf_rollouts = get_hp("num_reinf_rollouts")
        num_perf_rollouts = get_hp("num_perf_rollouts")
        gamma = get_hp("gamma")

        self._reinforce_rules_in_indiv(indiv, num_reinf_rollouts, gamma)
        self._assess_indiv_perf(indiv, num_perf_rollouts, gamma)

        # Return the modified Indiv obj. since this method is being
        # executed in other process via multiprocessing Pool and needs to
        # return modified obj. back to the main process.
        return indiv

    def _reinforce_rules_in_indiv(self, indiv, num_reinf_rollouts, gamma):
        # copy then reseed iod rng for reinf env so each indiv has own seeded
        # seq. of reinf trajectories and state of transition probs in env not
        # mutated between indivs, therefore gives same result for diff. num. of
        # CPUs used.
        reinf_env = copy.deepcopy(self._reinf_env)
        reinf_env.reseed_iod_rng(new_seed=indiv.id)

        # Sample a trajectory then reinforce it one-at-a-time
        for _ in range(num_reinf_rollouts):
            trajectory = self._gen_trajectory_using_indiv(reinf_env, indiv)
            self._reinforce_trajectory(trajectory, gamma)

    def _gen_trajectory_using_indiv(self, reinf_env, indiv):
        trajectory = []
        obs = reinf_env.reset()
        while not reinf_env.is_terminal():
            # do whole inference process here, i.e. no policy caching even if
            # indiv has it enabled. this is because the policy is mutating each
            # trajectory generated so *probably* not worth it
            (action, action_set) = infer_action_and_action_set(indiv, obs)
            if action != NULL_ACTION:
                assert action_set is not None
                reinf_env_response = reinf_env.step(action)
                reward = reinf_env_response.reward
                trajectory.append(
                    TrajectoryStep(obs, action, action_set, reward))
                obs = reinf_env_response.obs
            else:
                # trajectory is truncated
                assert action_set is None
                break
        return trajectory

    def _reinforce_trajectory(self, trajectory, gamma):
        t = len(trajectory)
        # iterate backwards over trajectory to incrementally calc payoffs for
        # action sets
        reward_sum = 0
        for i in range(t - 1, 0 - 1, -1):
            (obs, _, action_set, reward) = trajectory[i]
            reward_sum += reward
            steps_from_end = (t - 1 - i)
            payoff = (gamma**(steps_from_end)) * reward_sum
            update_action_set(action_set, payoff, obs)

    def _assess_indiv_perf(self, indiv, num_perf_rollouts, gamma):
        indiv.perf_assessment_res = assess_perf(self._perf_env, indiv,
                                                num_perf_rollouts, gamma)
