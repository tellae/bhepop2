"""
Simple uniform enrichment using a global distribution.
"""

from .base import SyntheticPopulationEnrichment
from bhepop2.sources.global_distribution import QuantitativeGlobalDistribution

import numpy as np


class SimpleUniformEnrichment(SyntheticPopulationEnrichment):
    """
    This class implements a simple enrichment using a global distribution.

    The global distribution describes the feature values of the whole population,
    using deciles (see GlobalDistribution).

    To evaluate a feature value for an individual, we randomly choose one of the deciles,
    and then draw a random value between its two boundaries.

    This method ensures a good distribution of the feature values
    over the total population, but no more.
    """

    def _assign_features(self):
        self.population[self.feature_name] = [
            self._draw_feature_value() for _ in range(len(self.population))
        ]

        return self.population

    def _draw_feature_value(self):
        rng = np.random.RandomState(self.seed)
        feature_index = rng.randint(len(self.source.feature_values))
        return self.source.get_value_for_feature(feature_index)

    def _validate_and_process_inputs(self):
        assert isinstance(self.source, QuantitativeGlobalDistribution)
