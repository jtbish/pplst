class Condition:
    def __init__(self, alleles, encoding):
        self._alleles = list(alleles)
        self._encoding = encoding
        # cache the phenotype
        self._phenotype = self._encoding.decode(self._alleles)
        self._matching_idx_order = \
            self._calc_matching_idx_order(self._phenotype)

    @property
    def alleles(self):
        return self._alleles

    @property
    def phenotype(self):
        return self._phenotype

    def _calc_matching_idx_order(self, phenotype):
        # first calc spans of all intervals
        spans_with_idxs = list(
            enumerate([interval.span for interval in phenotype]))
        # then sort the spans in ascending order
        sorted_spans_with_idxs = sorted(spans_with_idxs,
                                        key=lambda tup: tup[1],
                                        reverse=False)
        matching_idx_order = [tup[0] for tup in sorted_spans_with_idxs]
        assert len(matching_idx_order) == len(phenotype)
        return matching_idx_order

    def does_match(self, obs):
        for idx in self._matching_idx_order:
            interval = self._phenotype[idx]
            obs_val = obs[idx]
            if not interval.contains_val(obs_val):
                return False
        return True

    def __str__(self):
        return " && ".join([str(interval) for interval in self._phenotype])

    def __len__(self):
        return len(self._phenotype)
