import random
import pandas as pd
from bhepop2 import functions
from bhepop2.enrichment import Bhepop2Enrichment


class QualitativeEnrichment(Bhepop2Enrichment):
    """
    A class for enriching population using entropy maximisation.

    Notations used in this class documentation:

    - :math:`M_{k}` : crossed modality k (combination of attribute modalities)
    - :math:`F_{i}` : feature value i
    """

    mode = "qualitative"

    def _evaluate_feature_values(self):
        return functions.get_feature_from_qualitative_distribution(self.distributions)

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

    # Modification by POV

    def _compute_feature_prob(self, attribute, modality):
        """
        Compute the probability of being in each feature interval with the given modality.

        :param attribute: attribute name
        :param modality: attribute modality

        :return: DataFrame
        """

        prob_df = self.distributions[
            self.distributions["modality"].isin([modality])
            & self.distributions["attribute"].isin([attribute])
        ]

        res = pd.DataFrame({"feature": self.feature_values})
        res["prob"] = res["feature"].apply(lambda x: prob_df[x])

        return res

    def _draw_feature(self, res, index):
        """
        Draw a feature value using the given distribution.

        :param res: feature distributions by crossed modality
        :param index: index of the crossed modality

        :return: Drawn feature value
        """

        # get probs
        probs = res.loc[index,].to_numpy()
        # interval_values = [self.parameters["abs_minimum"]] + self.feature_values POV

        # get the non-null probs and values
        probs2 = []
        values2 = []
        for i in range(len(probs)):
            if pd.isna(probs[i]):
                continue
            probs2.append(probs[i])
            values2.append(self.feature_values[i])
        #    values2.append(interval_values[i]) POV

        # TODO : probs don't sum to 1, this isn't reassuring

        # draw a feature interval using the probs
        # values = list(range(len(values2))) POV
        final = random.choices(values2, probs2)[0]
        # feature_interval = random.choices(values, probs2)[0]

        # draw a feature value using a uniform distribution in the interval POV
        # lower, upper = interval_values[feature_interval], interval_values[feature_interval + 1]
        # draw = random.random()
        # final = lower + round((upper - lower) * draw)

        return final

    def _compute_feature_probabilities_from_distributions(self):
        """
        Compute the probability of each feature interval.

        Use the global distribution of the features to interpolate the interval probabilities.

        The resulting DataFrame contains :math:`P(f < F_{i})` for i in N

        :return: DataFrame
        """
        distrib_all_df = self.distributions[self.distributions["attribute"] == "all"]

        assert len(distrib_all_df) == 1

        res = pd.DataFrame({"feature": self.feature_values})
        res["prob"] = res["feature"].apply(lambda x: distrib_all_df[x])

        return res
