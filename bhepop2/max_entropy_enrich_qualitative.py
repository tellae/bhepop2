import logging as lg
import random
import pandas as pd
from bhepop2 import functions
from bhepop2.bhepop2_enrichment import Bhepop2Enrichment
import numpy as np


############  New class wich inherits from Bhepop2Enrichment POV ###############


class QualitativeEnrichment(Bhepop2Enrichment):
    """
    A class for enriching population using entropy maximisation.

    Notations used in this class documentation:

    - :math:`M_{k}` : crossed modality k (combination of attribute modalities)
    - :math:`F_{i}` : feature value i
    """

    #: json schema of the enrichment parameters Modified by POV
    parameters_schema = {
        "title": "QualitativeEnrichment parameters",
        "description": "Parameters of a population enrichment run",
        "type": "object",
        "properties": {
        },
    }

    def _init_distributions(self, distributions, attribute_selection):
        """
        Validate and filter the input distributions.

        When done, set the *distributions* field.

        :param distributions: input distributions DataFrame
        :param attribute_selection: distribution attributes selection
        """

        self.log("Setup distributions data")

        # validate distributions format and contents
        # functions.validate_distributions(distributions) POV  Pas de validation pour le moment

        distributions = distributions.copy()

        # filter distributions using the attribute selection
        if attribute_selection is not None:
            distributions = distributions[
                distributions["attribute"].isin(attribute_selection + ["all"])
            ]
            assert set(distributions["attribute"]) == set(
                attribute_selection + ["all"]
            ), "Mismatch between distribution attributes and attribute selection"

        # set distributions
        self.distributions = distributions

        # infer attributes and their modalities from the filtered distribution
        self.modalities = functions.infer_modalities_from_distributions(distributions)

    # MODIFIED BY POV
    def optimise(self):
        # compute crossed modalities frequencies
        self.log("Computing frequencies of crossed modalities", lg.INFO)
        self.crossed_modalities_frequencies = functions.compute_crossed_modalities_frequencies(
            self.population, self.modalities
        )
        self.log(
            "Number of crossed modalities present in the population: {}".format(
                len(self.crossed_modalities_frequencies)
            )
        )

        # compute vector of feature values
        self.log("Computing vector of all feature values", lg.INFO)
        self.feature_values = ["0voit", "1voit", "2voit", "3voit"]  # Modification léo
        # self.feature_values = functions.compute_feature_values(self.distributions) # Modification POV
        self.nb_features = len(self.feature_values)
        self.log("Number of feature values: {}".format(self.nb_features))

        # compute matrix of constraints
        self.log("Computing optimization constraints", lg.INFO)
        self._compute_constraints()

        # run resolution
        self.log("Starting optimization by entropy maximisation", lg.INFO)
        self.optim_result = self._run_optimization()

        return self.optim_result

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

        # modif Léo
        res = pd.DataFrame({"feature": self.feature_values})
        res["prob"] = res["feature"].apply(lambda x: prob_df[x])

        return res

    def assign_feature_value_to_pop(self):
        """
        Assign feature values to the population individuals using the algorithm results.

        :return: enriched population DataFrame
        """

        self.log("Drawing feature values for the population", lg.INFO)

        # compute the probability of being in each feature interval, for each crossed modality
        res = self._get_feature_probs()

        # associate each individual to a crossed modality
        self.crossed_modalities_frequencies["index"] = self.crossed_modalities_frequencies.index
        merge = self.population.merge(
            self.crossed_modalities_frequencies,
            how="left",
            on=functions.get_attributes(self.modalities),
        )

        # associate a feature value to the population individuals
        merge["feature"] = merge["index"].apply(lambda x: self._draw_feature(res, x))

        # remove irrelevant columns
        merge.drop(["index", "probability"], axis=1, inplace=True)

        return merge

    def _get_feature_probs(self):
        """
        For each crossed modality, compute the probability of belonging to a feature interval.

        Invert the crossed modality probabilities using Bayes.

        Compute

        .. math::

            P(f < F_{i} \mid M_{k}) = P(M_{k} \mid f < F_{i}) \\cdot \\frac{P(f < F_{i})}{P(M_{k})}

        :return: DataFrame
        """

        feature_probs = self._compute_feature_probabilities_from_distributions()

        res = self.optim_result

        nb_columns = len(res.columns)

        for c in res.columns:
            res[c] = res[c].map(lambda x: x[0])  # POV sur conseils de Valentin

        for i in range(nb_columns):
            res[i] = res[i] * feature_probs["prob"][i]

        for i in range(len(res)):
            total = res.iloc[i, :].sum()
            res.iloc[i, :] = res.iloc[i, :] / total

        pd.set_option("display.max_rows", 500)
        pd.set_option("display.max_columns", 500)
        pd.set_option("display.width", 1000)

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
        # modif Léo
        res = pd.DataFrame({"feature": self.feature_values})
        res["prob"] = res["feature"].apply(lambda x: distrib_all_df[x])

        return res

