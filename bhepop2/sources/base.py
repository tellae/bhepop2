"""
This module contains the abstract classes describing an enrichment source.
"""

from abc import ABC, abstractmethod
from bhepop2.utils import Bhepop2Logger, lg

DEFAULT_SOURCE_NAME = "Enrichment source"


class EnrichmentSource(ABC, Bhepop2Logger):
    """
    EnrichmentSource classes are supposed to provide ways
    to enrich or analyze a population.

    Sources describe a specific feature, for instance *declared income*.
    """

    def __init__(self, data, name: str = None):
        """
        Store the source data.

        Possible values of the described feature are evaluated from the source data.

        :param data: source data
        :param name: name of the source, used in displays such as analysis plots and tables
        """

        # init logging class
        Bhepop2Logger.__init__(self)

        self.name: str = DEFAULT_SOURCE_NAME if name is None else name

        self.data = data

        self._feature_values = None
        self._nb_feature_values = None

        self._validate_data()

    @property
    def feature_values(self):
        if self._feature_values is None:
            self.log("Computing vector of all feature values", lg.INFO)
            self._feature_values = self._evaluate_feature_values()
            self._nb_feature_values = len(self._feature_values)
        return self._feature_values

    @property
    def nb_feature_values(self):
        return self._nb_feature_values

    @abstractmethod
    def _evaluate_feature_values(self):
        """
        Evaluate the values that can be taken by the described feature.

        The result will be stored in the feature_values property.

        :return: object describing the feature values
        """
        pass

    @abstractmethod
    def _validate_data(self):
        """
        Validate the source data.

        Raise a ValueError if data is invalid.

        :raises: ValueError
        """
        pass

    @abstractmethod
    def get_value_for_feature(self, feature_index, rng):
        """
        Return a feature value for the given feature index.

        Generate a singular value from the feature state
        corresponding to the given index.

        :param feature_index: index of the feature in self.feature_values
        :param rng: Numpy random Generator

        :return: feature value
        """
        pass

    def compare_with_populations(self, populations, feature_name, **kwargs):
        """
        Compare the source data with populations containing the described feature (enriched or original)

        The class returns an instance of a PopulationAnalysis subclass, which
        can be used to generate different kinds of comparisons between the
        populations and the source data.

        :param populations: dict of populations {population_name: population}
        :param feature_name: population column containing the feature values
        :param kwargs: additional arguments for the analysis instance

        :return: PopulationAnalysis subclass instance.
        """
        raise NotImplementedError


class QuantitativeAttributes:
    """
    Class containing additional arguments used by quantitative sources.
    """

    def __init__(
        self,
        abs_minimum: int = 0,  # TODO: make it mandatory (no default value) ?
        relative_maximum: float = 1.5,
        delta_min: int = None,
    ):
        """
        Store parameters used for quantitative enrichment.

        :param abs_minimum: Minimum value of the feature distributions.
            This value is absolute, and thus equal for all distributions
        :param relative_maximum: Maximum value of the feature distributions.
            This value is relative and will be multiplied to the last value of each distribution
        :param delta_min: Minimum size of the feature intervals
        """

        self._abs_minimum = abs_minimum
        self._relative_maximum = relative_maximum
        self._delta_min = delta_min

        self._validate_quantitative_parameters()

    def _validate_quantitative_parameters(self):
        """
        Validate quantitative attributes' values.
        """
        assert self._relative_maximum >= 1
        assert self._delta_min is None or self._delta_min >= 0
