"""
This module contains base code for synthetic population enrichment classes.
"""

from bhepop2.utils import log, lg

from abc import ABC, abstractmethod
import pandas as pd
import random


class SyntheticPopulationEnrichment(ABC):
    """
    This abstract class describes the base attributes and methods of
    synthetic population enrichment classes.

    The class instances work on an original synthetic population,
    which is enriched using a dedicated algorithm.

    This enrichment process is executed in the assign_feature_value_to_pop method.
    Its implementation, and the algorithm used to evaluate the feature values,
    are core to the SyntheticPopulationEnrichment classes.
    """

    def __init__(self, population: pd.DataFrame, feature_name: str = None, seed=None):

        # random seed (maybe use a random generator instead)
        self.seed = seed
        if seed is not None:
            random.seed(seed)

        # original population DataFrame, to be enriched
        self.population: pd.DataFrame = population

        # name of the added column containing the new feature values
        feature_name = "feature" if feature_name is None else feature_name
        if feature_name in self.population.columns:
            raise ValueError(f"'{feature_name}' column already exists in population")
        self.feature_name: str = feature_name

        # list of possible feature values
        self._feature_values = None

        # number of feature values
        self._nb_features = None

        # resulting enriched population
        self.enriched_population = None

        # input validation
        self.log("Input data validation and preprocessing", lg.INFO)
        self._validate_and_process_inputs()

    @property
    def feature_values(self):
        if self._feature_values is None:
            self.log("Computing vector of all feature values", lg.INFO)
            self._feature_values = self._evaluate_feature_values()
            self._nb_features = len(self._feature_values)
        return self._feature_values

    @property
    def nb_features(self):
        return self._nb_features

    # feature assignment

    def assign_features(self):
        """
        Assign feature values to the population individuals.

        This method evaluates and adds feature values for each
        population individual.

        The name of the added column is defined by the feature_name class parameter.
        """

        self.enriched_population = self._assign_features()

        return self.enriched_population

    @abstractmethod
    def _assign_features(self):
        pass

    # validation and read

    @abstractmethod
    def _validate_and_process_inputs(self):
        pass

    @abstractmethod
    def _evaluate_feature_values(self):
        pass

    # analysis

    @abstractmethod
    def analyze_features(self):
        """
        Return or generate an analysis of the added features.

        :return:
        """
        pass

    # utils

    def log(message: str, level: int = lg.DEBUG):
        """
        Log a message using the package logger.

        See logging library.

        :param message: message to be logged
        :param level: logging level
        """

        log(message, level)

    log = staticmethod(log)
