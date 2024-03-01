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

    def __init__(
            self,
            population: pd.DataFrame,
            distributions,
            feature_name: str = None,
            seed=None,
    ):

        # random seed (maybe use a random generator instead)
        self.seed = seed
        if seed is not None:
            random.seed(seed)

        # original population DataFrame, to be enriched
        self.population: pd.DataFrame = population

        # enrichment data source
        self.distributions = distributions

        # name of the added column containing the new feature values
        feature_name = "feature" if feature_name is None else feature_name
        if feature_name in self.population.columns:
            raise ValueError(f"'{feature_name}' column already exists in population")
        self.feature_name: str = feature_name

        # resulting enriched population
        self.enriched_population = None

        # input validation
        self.log("Input data validation and preprocessing", lg.INFO)
        self._validate_and_process_inputs()

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

    # analysis

    def compare_with_distributions(self, enriched_population_name: str = "enriched_population", **kwargs):
        """
        Create a PopulationAnalysis instance for the enriched population.

        :param enriched_population_name: display name of the enriched population
        :param kwargs: additional parameters for the PopulationAnalysis instanciation

        :return: PopulationAnalysis for the current enriched population
        """
        if self.enriched_population is None:
            raise ValueError("No enriched population to analyze")

        return self.distributions.compare_with_populations(
            {enriched_population_name: self.enriched_population},
            self.feature_name,
            **kwargs
        )

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


class QuantitativeAttributes:

    def __init__(
            self,
            abs_minimum: int = 0,
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

    def _validate_quantitative_parameters(self):
        """
        Validate quantitative attributes' values.
        """
        assert self._relative_maximum >= 1
        assert self._delta_min is None or self._delta_min >= 0
