"""
This module contains base code for synthetic population enrichment classes.
"""

from bhepop2.utils import Bhepop2Logger, lg

from abc import ABC, abstractmethod
import pandas as pd
from numpy import random
from bhepop2.sources.base import EnrichmentSource
from bhepop2.utils import PopulationValidationError, SourceValidationError


class SyntheticPopulationEnrichment(ABC, Bhepop2Logger):
    """
    This abstract class describes the base attributes and methods of
    synthetic population enrichment classes.

    The class instances work on an original synthetic population,
    which is enriched using a dedicated algorithm.

    This enrichment process is executed in the assign_feature_value_to_pop method.
    Its implementation, and the algorithm used to evaluate the feature values,
    are core to the SyntheticPopulationEnrichment classes.
    """

    _required_source_class = EnrichmentSource

    def __init__(
        self,
        population: pd.DataFrame,
        source,
        feature_name: str = None,
        seed=None,
    ):
        # init logging class
        Bhepop2Logger.__init__(self)

        # original population DataFrame, to be enriched
        if population.empty:  # pragma: no cover
            raise PopulationValidationError("Population to enrich is empty")
        self.population: pd.DataFrame = population.copy()

        # enrichment data source
        if not isinstance(source, self._required_source_class):
            raise SourceValidationError(f"{self.__class__.__name__} requires an instance "
                                        f"of {self._required_source_class} as a source")
        self.source = source

        # name of the added column containing the new feature values
        feature_name = "feature" if feature_name is None else feature_name
        if feature_name in self.population.columns:
            raise ValueError(f"'{feature_name}' column already exists in population")
        self.feature_name: str = feature_name

        # random state
        self.seed = seed
        self.rng = random.default_rng(seed)

        # input validation
        self.log("Input data validation and preprocessing", lg.INFO)
        self._validate_and_process_inputs()
        self.source.usable_with_population(self.population)

    # feature assignment

    def assign_feature_values(self):
        """
        Assign feature values to the population individuals.

        This method evaluates and adds feature values to each
        population individual. The name of the added column is
        defined by the feature_name class parameter.

        Returned enriched population is a copy of the original population.
        The original population is not modified.

        :return: enriched population
        """

        self.population[self.feature_name] = self._evaluate_feature_on_population()

        return self.population

    @abstractmethod
    def _evaluate_feature_on_population(self):
        """
        Evaluate a list of feature values for each individual.

        :return: iterable with same size and order than the population
        """
        # implement feature evaluation using a dedicated algorithm
        raise NotImplementedError

    def _get_value_for_feature(self, feature_id):
        """
        Get a feature value for the given feature id.

        This method is a helper that class self.source.get_value_for_feature
        with feature id and self.rng.

        :param feature_id:

        :return: feature value
        """
        return self.source.get_value_for_feature(feature_id, self.rng)

    # validation and read

    def _validate_and_process_inputs(self):
        """
        Validate and process the provided enrichment inputs.

        Both the population and the enrichment source may need to be validated.

        :raise: ValueError if validation fails
        """
        pass

    # analysis

    def compare_with_source(self, enriched_population_name: str = "enriched_population", **kwargs):
        """
        Create a PopulationAnalysis instance for the enriched population.

        :param enriched_population_name: display name of the enriched population
        :param kwargs: additional parameters for the PopulationAnalysis instantiation

        :return: PopulationAnalysis for the current enriched population
        """

        return self.source.compare_with_populations(
            {enriched_population_name: self.population}, self.feature_name, **kwargs
        )
