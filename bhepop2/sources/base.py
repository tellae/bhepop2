"""
This module contains the abstract classes describing an enrichment source.
"""

from abc import ABC, abstractmethod
import logging as lg


class EnrichmentSource(ABC):
    """
    EnrichmentSource classes are supposed to provide ways
    to enrich or analyze a population.

    Sources describe a specific feature, for instance *declared income*.
    """

    def __init__(self, data):
        """
        Store the source data.

        Possible values of the described feature are evaluated from the source data.

        :param data: source data
        """

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
    def get_value_for_feature(self, feature_index):
        """
        Return a feature value for the given feature index.

        Generate a singular value from the feature state
        corresponding to the given index.

        :param feature_index: index of the feature in self.feature_values

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

    def log(self, message, level):
        pass

