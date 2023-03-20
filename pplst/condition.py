_SPAN_FRAC_MIN_INCL = 0
_SPAN_FRAC_MAX_INCL = 1


class Condition:
    def __init__(self, alleles, encoding):
        self._alleles = list(alleles)
        self._encoding = encoding
        # cache the phenotype
        self._phenotype = self._encoding.decode(self._alleles)
        self._matching_idx_order = \
            self._calc_matching_idx_order(self._phenotype,
                                          obs_space=self._encoding.obs_space)

    @property
    def alleles(self):
        return self._alleles

    @property
    def phenotype(self):
        return self._phenotype

    def _calc_matching_idx_order(self, phenotype, obs_space):
        # first calc "span fracs" of all intervals in phenotype relative to
        # each dim span
        assert len(phenotype) == len(obs_space)
        span_fracs_with_idxs = []
        for (idx, (interval, dim)) in enumerate(zip(phenotype, obs_space)):
            span_frac = (interval.span / dim.span)
            assert _SPAN_FRAC_MIN_INCL <= span_frac <= _SPAN_FRAC_MAX_INCL
            span_fracs_with_idxs.append((idx, span_frac))

        # then sort the span fracs in ascending order
        sorted_span_fracs_with_idxs = sorted(span_fracs_with_idxs,
                                             key=lambda tup: tup[1],
                                             reverse=False)
        matching_idx_order = [tup[0] for tup in sorted_span_fracs_with_idxs]
        assert len(matching_idx_order) == len(phenotype)
        return matching_idx_order

    def does_match(self, obs):
        for idx in self._matching_idx_order:
            interval = self._phenotype[idx]
            obs_val = obs[idx]
            if not interval.contains_val(obs_val):
                return False
        return True

    def __eq__(self, other):
        # encoding must logically be the same implicitly so don't bother to
        # check.
        # encoding generates phenotype and matching idx order so also don't
        # check as implicitly the same
        return self._alleles == other._alleles

    def __len__(self):
        return len(self._phenotype)

    def __str__(self):
        return " && ".join([str(interval) for interval in self._phenotype])
