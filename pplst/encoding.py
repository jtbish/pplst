import abc
import math

from rlenvs.obs_space import IntegerObsSpace, RealObsSpace

from .condition import Condition
from .hyperparams import get_hyperparam as get_hp
from .interval import IntegerInterval, RealInterval
from .rng import get_rng

_GENERALITY_UB_INCL = 1.0


class EncodingABC(metaclass=abc.ABCMeta):
    def __init__(self, obs_space):
        self._obs_space = obs_space

    @property
    def obs_space(self):
        return self._obs_space

    @abc.abstractmethod
    def init_condition(self):
        raise NotImplementedError

    @abc.abstractmethod
    def decode(self, cond_alleles):
        raise NotImplementedError

    @abc.abstractmethod
    def calc_condition_generality(self, cond_intervals):
        raise NotImplementedError

    @abc.abstractmethod
    def mutate_condition_alleles(self, cond_alleles):
        raise NotImplementedError


class UnorderedBoundEncodingABC(EncodingABC, metaclass=abc.ABCMeta):
    def init_condition(self):
        num_alleles = (len(self._obs_space) * 2)
        cond_alleles = []
        for dim in self._obs_space:
            (lower, upper) = self._init_alleles_for_dim(dim)
            dim_alleles = [lower, upper]
            # to avoid bias, insert alleles into genotype in random order
            get_rng().shuffle(dim_alleles)
            for allele in dim_alleles:
                cond_alleles.append(allele)
        assert len(cond_alleles) == num_alleles
        return Condition(cond_alleles, self)

    @abc.abstractmethod
    def _init_alleles_for_dim(self, dim):
        """Return (lower, upper) init alleles, with lower <= upper."""
        raise NotImplementedError

    def decode(self, cond_alleles):
        phenotype = []
        assert len(cond_alleles) % 2 == 0
        for i in range(0, len(cond_alleles), 2):
            first_allele = cond_alleles[i]
            second_allele = cond_alleles[i + 1]
            lower = min(first_allele, second_allele)
            upper = max(first_allele, second_allele)
            phenotype.append(self._INTERVAL_CLS(lower, upper))
        assert len(phenotype) == len(cond_alleles) // 2
        return phenotype

    @abc.abstractmethod
    def calc_condition_generality(self, cond_intervals):
        raise NotImplementedError

    def mutate_condition_alleles(self, alleles):
        assert len(alleles) % 2 == 0
        # iterate in allele pairs (i.e. over each dim)
        allele_pairs = [(alleles[i], alleles[i + 1])
                        for i in range(0, len(alleles), 2)]
        mut_alleles = []
        for (allele_pair, dim) in zip(allele_pairs, self._obs_space):
            for allele in allele_pair:
                if get_rng().random() < get_hp("p_mut"):
                    noise = self._gen_mutation_noise(dim)
                    mut_allele = (allele + noise)
                    mut_allele = max(mut_allele, dim.lower)
                    mut_allele = min(mut_allele, dim.upper)
                    mut_alleles.append(mut_allele)
                else:
                    mut_alleles.append(allele)
        assert len(mut_alleles) == len(alleles)
        return mut_alleles

    @abc.abstractmethod
    def _gen_mutation_noise(self, dim):
        """Mutation noise, *inclusive of sign*"""
        raise NotImplementedError


class IntegerUnorderedBoundEncoding(UnorderedBoundEncodingABC):
    _GENERALITY_LB_EXCL = 0
    _INTERVAL_CLS = IntegerInterval
    _GEOM_MUT_TARGET_MASS = 0.99

    def __init__(self, obs_space):
        assert isinstance(obs_space, IntegerObsSpace)
        super().__init__(obs_space)

    def _init_alleles_for_dim(self, dim):
        r_nought = get_hp("r_nought")
        # rand integer ~ [0, r_nought] for lower, upper
        first = get_rng().randint(low=0, high=(r_nought + 1))
        second = get_rng().randint(low=0, high=(r_nought + 1))
        lower = min(first, second)
        upper = max(first, second)
        assert lower <= upper
        return (lower, upper)

    def calc_condition_generality(self, cond_intervals):
        # condition generality calc as in
        # Wilson '00 Mining Oblique Data with XCS
        numer = sum([interval.span for interval in cond_intervals])
        denom = sum([dim.span for dim in self._obs_space])
        generality = numer / denom
        assert self._GENERALITY_LB_EXCL < generality <= _GENERALITY_UB_INCL
        return generality

    def _gen_mutation_noise(self, dim):
        """'Dimension aware' geometric mutation."""
        # base noise is integer ~ Geo(p): supported on integers >= 1 i.e.
        # "shifted" geom. dist.
        # set p for geom dist according to satisfying target prob.
        # mass on CDF after k trials, k = floor(dim.span / 2), i.e. satisfy
        # target mass over half dim span
        k = math.floor(dim.span / 2)
        # rearranged CDF eqn. to solve for p
        p = 1 - (1 - self._GEOM_MUT_TARGET_MASS)**(1 / k)
        geom_noise = get_rng().geometric(p)
        sign = get_rng().choice([-1, 1])
        return (sign * geom_noise)


class RealUnorderedBoundEncoding(UnorderedBoundEncodingABC):
    _GENERALITY_LB_INCL = 0
    _INTERVAL_CLS = RealInterval
    _MUT_MEAN = 0.0

    def __init__(self, obs_space):
        assert isinstance(obs_space, RealObsSpace)
        super().__init__(obs_space)

    def _init_alleles_for_dim(self, dim):
        center = get_rng().uniform(low=dim.lower, high=dim.upper)
        # for continuous space, r_nought interpreted as fraction of dim span
        # over which to draw noise from for span
        r_nought = get_hp("r_nought")
        assert 0.0 < r_nought <= 1.0
        spread_high = (r_nought * dim.span)
        spread = get_rng().uniform(low=0, high=spread_high)
        lower = (center - spread)
        upper = (center + spread)
        # trunc
        lower = max(lower, dim.lower)
        upper = min(upper, dim.upper)
        assert lower <= upper
        return (lower, upper)

    def calc_condition_generality(self, cond_intervals):
        numer = sum([interval.span for interval in cond_intervals])
        denom = sum([dim.span for dim in self._obs_space])
        generality = numer / denom
        assert self._GENERALITY_LB_INCL <= generality <= _GENERALITY_UB_INCL
        return generality

    def _gen_mutation_noise(self, dim):
        """For reals, mutation is Gaussian noise, mean=0, stdev dependent on
        magnitude of dim operating on."""
        stdev = (get_hp("mut_sigma_pcnt") * dim.span)
        return get_rng().normal(loc=self._MUT_MEAN, scale=stdev)
