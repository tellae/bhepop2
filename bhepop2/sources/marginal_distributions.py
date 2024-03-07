"""
This module contains classes describing marginal distributions sources.

In this scope, specific source distributions are known for population subsets.
This allows a more precise feature value association than a global, population wide distribution.
"""

from .base import EnrichmentSource, QuantitativeAttributes
from bhepop2 import functions
from bhepop2.analysis import QuantitativeAnalysis, QualitativeAnalysis

import pandas as pd
from abc import abstractmethod

#: attribute and modality corresponding to the global distribution
ALL_LABEL = "all"


class MarginalDistributions(EnrichmentSource):
    """
    Abstract class describing marginal distributions source.

    In this class, the distributions subsets are known
    for population individuals presenting a specific attribute.
    For instance, the Filosofi data source (INSEE) stores distributions of
    declared income in administrative areas, for the whole population and for
    population subsets, such as tenants or owners.

    In this scope, we use the following terms to describe such marginal distributions:

    - An **attribute** refers to an information in the initial sample or in the aggregate data.
        For instance: age, profession, ownership, etc.
    - **Modalities** are the partition of one attribute.
        For instance, in Filosofi, the *ownership* attribute can take the values *Owner* and *Tenant*.
    - **Cross modalities** are the intersection of two or more modalities.
        For instance, *Owner* and *above 65 years old*.


    Then, population individuals are part of a single cross modality,
    and can be matched with distributions corresponding to their known attributes.
    """

    def __init__(self, data, name=None, attribute_selection: list = None):
        """
        Store modality distributions and attribute selection.

        Modality distributions come as a DataFrame with *attribute* and *modality*
        columns. The rest of the columns should describe the distribution associated
        to this modality.
        An additional row with attribute and modality equal to ALL_LABEL is
        expected to contain a distribution describing the global population.

        Attribute selection is used to indicate the distributions that will be used
        as an enrichment source. If no selection is provided, all attributes are used.

        :param data: DataFrame describing feature values distributions for each modality
        :param name: name of the enrichment source
        :param attribute_selection: distribution attributes used. By default, use all attributes of the distribution
        """
        # distribution attributes used for feature evaluation
        self.attribute_selection = attribute_selection

        # attributes considered for the assignment, with their modalities
        # { attribute: [modalities] }
        self.modalities = None

        super().__init__(data, name=name)

    def _validate_data(self):
        # quantitative or qualitative check
        self._validate_data_type()

        # filter the distributions to keep only those corresponding to the attribute selection
        # and infer the modalities from the remaining attributes
        self.data, self.modalities = functions.filter_distributions_and_infer_modalities(
            self.data, self.attribute_selection
        )

        # check that there are modalities at the end
        assert (
            len(self.modalities.keys()) > 0
        ), "No attributes found in distributions for enriching population"

    @abstractmethod
    def _validate_data_type(self):
        pass

    @abstractmethod
    def compute_feature_prob(self, attribute=ALL_LABEL, modality=ALL_LABEL):
        """
        Return a DataFrame containing the probability to be in each feature state while in the given modality.

        The resulting DataFrame is of the following format:
        { "feature": [feature_values], "prob": [feature_probs] }

        This method accepts attributes and modalities from self.modalities and
        also (ALL_LABEL, ALL_LABEL) couple, returning the global distribution.

        :param attribute: attribute label
        :param modality: modality label

        :return: DataFrame["feature", "prob"]
        """
        pass

    def get_modality_distribution(self, attribute, modality):
        """
        Get the distribution corresponding to the given attribute and modality.

        This method accepts attributes and modalities from self.modalities and
        also (ALL_LABEL, ALL_LABEL) couple, returning the global distribution.

        :param attribute: attribute label
        :param modality: modality label
        :return:
        """
        return self.data[
            self.data["modality"].isin([modality]) & self.data["attribute"].isin([attribute])
        ]


class QualitativeMarginalDistributions(MarginalDistributions):
    """
    Marginal distributions describing qualitative features.

    **Input data**:

    DataFrame with feature values as columns, and probabilities as
    column values, for each attribute/modality pair. An additional row containing
    a global distribution (for the whole population) must be present, with
    attribute and modality equal to :attr:`~bhepop2.sources.marginal_distributions.ALL_LABEL`.

    **Example**:

    .. list-table:: Table containing qualitative marginal distributions for attributes **ownership** and **age**
        :widths: 10 10 10 20 20
        :header-rows: 1

        * - Red
          - Green
          - Blue
          - attribute
          - modality
        * - 0.3
          - 0.3
          - 0.4
          - all
          - all
        * - 0.5
          - 0.2
          - 0.3
          - ownership
          - Owner
        * - 0.4
          - 0.4
          - 0.2
          - ownership
          - Tenant
        * - 0
          - 0.5
          - 0.5
          - age
          - 0_29
        * - ...
          - ...
          - ...
          - ...
          - ...
        * - 0.7
          - 0.1
          - 0.2
          - age
          - 75_or_more

    """

    def _evaluate_feature_values(self):
        """
        Evaluate the feature values from the distributions columns.

        :return: list of feature values
        """
        return functions.get_feature_from_qualitative_distribution(self.data)

    def _validate_data_type(self):
        # TODO : test that self._abs_minimum is inferior to all distribution values
        functions.validate_distributions(self.data, self.attribute_selection, "qualitative")

    def compute_feature_prob(self, attribute=ALL_LABEL, modality=ALL_LABEL):
        # get distribution for the given modality
        prob_df = self.get_modality_distribution(attribute, modality)

        # get probabilities with direct application of the distributions
        res = pd.DataFrame({"feature": self.feature_values})
        res["prob"] = res["feature"].apply(lambda x: prob_df[x])

        return res

    def get_value_for_feature(self, feature_index, rng):
        # directly return the stored feature value
        return self.feature_values[feature_index]

    def compare_with_populations(self, populations, feature_name, **kwargs):
        return QualitativeAnalysis(
            populations=populations,
            modalities=self.modalities,
            feature_column=feature_name,
            distributions=self.data,
            distributions_name=self.name,
            **kwargs
        )


class QuantitativeMarginalDistributions(MarginalDistributions, QuantitativeAttributes):
    """
    Marginal distributions describing quantitative features.

    **Input data**:

    DataFrame with deciles numbers as columns (D1, D2 to D9),
    and values as column values, for each attribute/modality pair. An additional row containing
    a global distribution (for the whole population) must be present, with
    attribute and modality equal to :attr:`~bhepop2.sources.marginal_distributions.ALL_LABEL`.

    **Example**:

    .. list-table:: Table containing quantitative marginal distributions for attributes **ownership** and **age**
        :widths: 25 10 25 40 40
        :header-rows: 1

        * - D1
          - ...
          - D9
          - attribute
          - modality
        * - 18 852
          - ...
          - 46 522
          - all
          - all
        * - 16 542
          - ...
          - 50 060
          - ownership
          - Owner
        * - 8 764
          - ...
          - 29 860
          - ownership
          - Tenant
        * - 15 000
          - ...
          - 45 000
          - age
          - 0_29
        * - ...
          - ...
          - ...
          - ...
          - ...
        * - 20 000
          - ...
          - 65 000
          - age
          - 75_or_more

    """

    def __init__(
        self,
        data,
        name=None,
        attribute_selection: list = None,
        abs_minimum: int = 0,
        relative_maximum: float = 1.5,
        delta_min: int = None,
    ):

        QuantitativeAttributes.__init__(
            self,
            abs_minimum,
            relative_maximum,
            delta_min,
        )

        MarginalDistributions.__init__(
            self,
            data,
            name=name,
            attribute_selection=attribute_selection,
        )

    def _evaluate_feature_values(self):
        """
        Evaluate the feature values from the distribution values and class parameters.

        :return: list of feature values
        """
        return functions.compute_feature_values(self.data, self._relative_maximum, self._delta_min)

    def _validate_data_type(self):
        functions.validate_distributions(self.data, self.attribute_selection, "quantitative")

    def compute_feature_prob(self, attribute=ALL_LABEL, modality=ALL_LABEL):
        # get distribution for the given modality
        decile_tmp = self.get_modality_distribution(attribute, modality)

        # interpolate feature probabilities from distributions
        total_population_decile_tmp = [
            self._abs_minimum,
            float(decile_tmp["D1"].iloc[0]),
            float(decile_tmp["D2"].iloc[0]),
            float(decile_tmp["D3"].iloc[0]),
            float(decile_tmp["D4"].iloc[0]),
            float(decile_tmp["D5"].iloc[0]),
            float(decile_tmp["D6"].iloc[0]),
            float(decile_tmp["D7"].iloc[0]),
            float(decile_tmp["D8"].iloc[0]),
            float(decile_tmp["D9"].iloc[0]),
            self.feature_values[-1],
        ]
        prob_df = functions.compute_features_prob(self.feature_values, total_population_decile_tmp)

        return prob_df

    def get_value_for_feature(self, feature_index, rng):
        """
        Return a value drawn from the interval corresponding to the feature index.

        The first interval is defined as [self._abs_minimum, self.feature_values[0]].
        and so on. The value is drawn using a uniform rule.

        :param feature_index:
        :param rng:
        :return:
        """

        interval_values = [self._abs_minimum] + self.feature_values

        lower, upper = interval_values[feature_index], interval_values[feature_index + 1]

        draw = rng.uniform()

        drawn_feature_value = lower + (upper - lower) * draw

        return drawn_feature_value

    def compare_with_populations(self, populations, feature_name, **kwargs):
        return QuantitativeAnalysis(
            populations=populations,
            modalities=self.modalities,
            feature_column=feature_name,
            distributions=self.data,
            distributions_name=self.name,
            **kwargs
        )
