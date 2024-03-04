
from abc import ABC, abstractmethod
import logging as lg


class EnrichmentSource(ABC):

    def __init__(self, data):

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
        pass

    @abstractmethod
    def _validate_data(self):
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
        raise NotImplementedError

    def log(self, message, level):
        pass

