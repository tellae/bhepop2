"""
Simple uniform enrichment using a global distribution.

This enrichment method is not very interesting, but it provides a simple
implementation of the enrichment logic. It also provides a good comparison
point with other enrichment methods.
"""

from .base import SyntheticPopulationEnrichment
from bhepop2.sources.global_distribution import QuantitativeGlobalDistribution


class SimpleUniformEnrichment(SyntheticPopulationEnrichment):
    """
    This class implements a simple enrichment using a global distribution.

    **Expected source types**:

    .. autosummary::
        :nosignatures:

        ~bhepop2.sources.global_distribution.QuantitativeGlobalDistribution

    ------

    The global distribution describes the feature values of the whole population,
    using deciles (see :mod:`~bhepop2.sources.global_distribution`).

    To evaluate a feature value for an individual, we randomly choose one of the deciles,
    and then draw a random value between its two boundaries.

    This method ensures a good distribution of the feature values
    over the total population, but no more.
    """

    def _evaluate_feature_on_population(self):
        feature_values = [self._draw_feature_value() for _ in range(len(self.population))]

        return feature_values

    def _draw_feature_value(self):
        feature_index = self.rng.integers(len(self.source.feature_values))
        return self._get_value_for_feature(feature_index)

    def _validate_and_process_inputs(self):
        assert isinstance(self.source, QuantitativeGlobalDistribution)
