
from .base import EnrichmentSource, QuantitativeAttributes
import random
import numpy as np


class QuantitativeGlobalDistribution(EnrichmentSource, QuantitativeAttributes):
    """
    This class describes a global distribution used as an enrichment source.

    For now, only deciles distributions are managed.
    """

    def __init__(self, data, name=None, abs_minimum: int = 0,
        relative_maximum: float = 1.5):

        EnrichmentSource.__init__(self, data, name=name)

        QuantitativeAttributes.__init__(self, abs_minimum=abs_minimum, relative_maximum=relative_maximum)

    def _evaluate_feature_values(self):
        """
        Directly return the deciles values, plus the last one multiplied by _relative_maximum.

        :return: list of feature values
        """
        values = list(self.data[["D1", "D2", "D3", "D4", "D5", "D6", "D7", "D8", "D9"]].iloc[0].values)
        values.append(np.max(values) * self._relative_maximum)

        return values

    def _validate_data(self):
        """
        Check that the deciles columns are present and that length is 1.
        """
        assert set(self.data.columns) >= {f"D{i}" for i in range(1, 10)}
        assert len(self.data) == 1

    def get_value_for_feature(self, feature_index):
        """
        Return a value drawn from the interval corresponding to the feature index.

        The first interval is defined as [self._abs_minimum, self.feature_values[0]].
        and so on. The value is drawn using a uniform rule.

        :param feature_index:
        :return:
        """

        interval_values = [self._abs_minimum] + self.feature_values

        lower, upper = interval_values[feature_index], interval_values[feature_index + 1]

        draw = random.random()

        # TODO : pas d'arrondi
        drawn_feature_value = round(
            lower + (upper - lower) * draw
        )

        return drawn_feature_value

