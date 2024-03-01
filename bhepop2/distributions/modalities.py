from .base import Distributions, ALL_LABEL
from bhepop2 import functions

import pandas as pd
from abc import abstractmethod
from bhepop2.enrichment.base import QuantitativeAttributes
from bhepop2.analysis import QuantitativeAnalysis, QualitativeAnalysis
import random


class ModalitiesDistributions(Distributions):
    """

    """

    def __init__(self, data, attribute_selection: list = None):
        """
        Store modality distributions and attribute selection.

        :param data: DataFrame describing feature values distributions for each modality
        :param attribute_selection: distribution attributes used. By default, use all attributes of the distribution
        """
        # distribution attributes used for feature evaluation
        self.attribute_selection = attribute_selection

        # attributes considered for the assignment, with their modalities
        # { attribute: [modalities] }
        self.modalities = None

        super().__init__(data)

    def _validate_data(self):

        # TODO : qualitative or quantitative
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
            self.data["modality"].isin([modality])
            & self.data["attribute"].isin([attribute])
        ]


class QualitativeModalitiesDistributions(ModalitiesDistributions):

    def _evaluate_feature_values(self):
        """
        Evaluate the feature values from the distributions columns.

        :return: list of feature values
        """
        return functions.get_feature_from_qualitative_distribution(self.data)

    def _validate_data_type(self):
        functions.validate_distributions(self.data, self.attribute_selection, "qualitative")

    def compute_feature_prob(self, attribute=ALL_LABEL, modality=ALL_LABEL):
        # get distribution for the given modality
        prob_df = self.get_modality_distribution(attribute, modality)
        print(prob_df)
        # get probabilities with direct application of the distributions
        res = pd.DataFrame({"feature": self.feature_values})
        res["prob"] = res["feature"].apply(lambda x: prob_df[x])

        return res

    def get_value_for_feature(self, feature_index):
        # directly return the stored feature value
        return self.feature_values[feature_index]

    def compare_with_populations(self, populations, feature_name, **kwargs):
        return QualitativeAnalysis(
            populations=populations,
            modalities=self.modalities,
            feature_column=feature_name,
            distributions=self.data,
            **kwargs
        )


class QuantitativeModalitiesDistributions(ModalitiesDistributions, QuantitativeAttributes):

    def __init__(
            self,
            data,
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

        ModalitiesDistributions.__init__(
            self,
            data,
            attribute_selection,
        )

    def _evaluate_feature_values(self):
        """
        Evaluate the feature values from the distribution values and class parameters.

        :return: list of feature values
        """
        return functions.compute_feature_values(
            self.data, self._relative_maximum, self._delta_min
        )

    def _validate_data_type(self):
        functions.validate_distributions(self.data, self.attribute_selection, "quantitative")

    def compute_feature_prob(self, attribute=ALL_LABEL, modality=ALL_LABEL):
        # get distribution for the given modality
        decile_tmp = self.get_modality_distribution(attribute, modality)

        # interpolate feature probabilities from distributions
        total_population_decile_tmp = [
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

    def get_value_for_feature(self, feature_index):
        """
        Return a value drawn from the interval corresponding to the feature index.

        The first interval is defined as [self._abs_minimum, self.feature_values[0]].
        and so on. The value is drawn using a uniform rule.

        :param feature_index:
        :return:
        """

        interval_values = [self._abs_minimum] + self.feature_values

        lower, upper = interval_values[feature_index - 1], interval_values[feature_index]

        draw = random.random()

        # TODO : pas d'arrondi
        drawn_feature_value = round(
            lower + (upper - lower) * draw
        )

        return drawn_feature_value

    def compare_with_populations(self, populations, feature_name, **kwargs):
        return QuantitativeAnalysis(
            populations=populations,
            modalities=self.modalities,
            feature_column=feature_name,
            distributions=self.data,
            **kwargs
        )
