import logging as lg
import numpy as np
import random
import pandas as pd
from bhepop2 import utils
from bhepop2 import functions
from .optim import minxent_gradient
from bhepop2.enrichment import Bhepop2Enrichment


class QuantitativeEnrichment(Bhepop2Enrichment):
    """
    A class for enriching population using entropy maximisation.

    Notations used in this class documentation:

    - :math:`M_{k}` : crossed modality k (combination of attribute modalities)

    - :math:`F_{i}` : feature class i

        - For quantitative features, corresponds to a numeric interval.
        - For qualitative features, corresponds to one of the feature values.
    """

    mode = "quantitative"

    #: json schema of the enrichment parameters
    parameters_schema = {
        "title": "Enrichment parameters",
        "description": "Parameters of a population enrichment run",
        "type": "object",
        "required": [
            "abs_minimum",
            "relative_maximum",
        ],
        "properties": {
            "abs_minimum": {
                "title": "Distributions absolute minimum",
                "description": "Minimum value of the feature distributions. This value is absolute, and thus equal for all distributions.",
                "type": "number",
                "default": 0,
            },
            "relative_maximum": {
                "title": "Distributions relative maximum",
                "description": "Maximum value of the feature distributions. This value is relative and will be multiplied to the last value of each distribution.",
                "type": "number",
                "default": 1.5,
                "minimum": 1,
            },
            "delta_min": {
                "title": "Minimum feature value delta",
                "description": "Minimum size of the feature intervals",
                "type": ["null", "number"],
                "default": None,
                "minimum": 0,
            },
        },
    }

    def _evaluate_feature_values(self):
        return functions.compute_feature_values(
            self.distributions, self.parameters["relative_maximum"], self.parameters["delta_min"]
        )

    def _init_distributions(self):
        """
        Validate and filter the input distributions.

        When done, set the *distributions* field.
        """

        self.log("Setup distributions data")

        # validate distributions format and contents
        functions.validate_distributions(self.distributions, self.attribute_selection, self.mode)

        # filter distributions and infer modalities
        self.distributions, self.modalities = functions.filter_distributions_and_infer_modalities(
            self.distributions, self.attribute_selection
        )

        # check that there are modalities at the end
        assert (
            len(self.modalities.keys()) > 0
        ), "No attributes found in distributions for enriching population"

    def _compute_feature_prob(self, attribute, modality):
        """
        Compute the probability of being in each feature interval with the given modality.

        :param attribute: attribute name
        :param modality: attribute modality

        :return: DataFrame
        """

        decile_tmp = self.distributions[
            self.distributions["modality"].isin([modality])
            & self.distributions["attribute"].isin([attribute])
        ]

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

    def _draw_feature(self, res, index):
        """
        Draw a feature value using the given distribution.

        :param res: feature distributions by crossed modality
        :param index: index of the crossed modality

        :return: Drawn feature value
        """

        # get probs
        probs = res.loc[index,].to_numpy()
        interval_values = [self.parameters["abs_minimum"]] + self.feature_values

        # get the non-null probs and values
        probs2 = []
        values2 = [self.parameters["abs_minimum"]]
        for i in range(len(probs)):
            if pd.isna(probs[i]):
                continue
            probs2.append(probs[i])
            values2.append(interval_values[i + 1])
        # TODO : probs don't sum to 1, this isn't reassuring

        # draw a feature interval using the probs
        values = list(range(len(probs2)))
        feature_interval = random.choices(values, probs2)[0]
        assert len(probs2) == len(values2) - 1

        # draw a feature value using a uniform distribution in the interval
        lower, upper = values2[feature_interval], values2[feature_interval + 1]

        draw = random.random()

        final = round(
            lower + (upper - lower) * draw
        )  # POV : pourquoi on arrondit ? Cela n'explique pas les problèmes

        return final

    def _compute_feature_probabilities_from_distributions(self):
        """
        Compute the probability of each feature interval.

        Use the global distribution of the features to interpolate the interval probabilities.

        The resulting DataFrame contains :math:`P(f \\in F_{i})` for i in N

        :return: DataFrame
        """
        distrib_all_df = self.distributions[self.distributions["attribute"] == "all"]

        assert len(distrib_all_df) == 1

        total_population_decile = [
            self.distributions["D1"].iloc[0],
            self.distributions["D2"].iloc[0],
            self.distributions["D3"].iloc[0],
            self.distributions["D4"].iloc[0],
            self.distributions["D5"].iloc[0],
            self.distributions["D6"].iloc[0],
            self.distributions["D7"].iloc[0],
            self.distributions["D8"].iloc[0],
            self.distributions["D9"].iloc[0],
            self.feature_values[-1],  # maximum value of all deciles
        ]

        prob_df = functions.compute_features_prob(self.feature_values, total_population_decile)

        return prob_df

