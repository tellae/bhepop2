import logging as lg
import numpy as np
import random
import pandas as pd
from bhepop2 import utils
from bhepop2 import functions
import math
from .optim import minxent_gradient


class Bhepop2Enrichment:
    """
    A class for enriching population using entropy maximisation.

    Notations used in this class documentation:

    - :math:`M_{k}` : crossed modality k (combination of attribute modalities)
    - :math:`F_{i}` : feature value i
    """

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

    def __init__(
        self,
        population: pd.DataFrame,
        distributions: pd.DataFrame,
        attribute_selection: list = None,
        parameters=None,
        seed=None,
    ):
        """
        Synthetic population enrichment class.

        :param population: enriched population
        :param distributions: enriching data distributions
        :param attribute_selection: distribution attributes used. By default, use all attributes of the distribution
        :param parameters: enrichment parameters
        :param seed: random seed
        """

        # random seed (maybe use a random generator instead)
        self.seed = seed
        if seed is not None:
            random.seed(seed)

        # original population to be enriched
        self.population = None

        # distributions of the feature values by modality
        self.distributions = None

        # attributes considered for the assignment, with their modalities
        # { attribute: [modalities] }
        self.modalities = None

        # execution parameters
        if parameters is None:
            parameters = dict()
        self.parameters = utils.add_defaults_and_validate_against_schema(
            parameters, self.parameters_schema
        )

        # algorithm data

        # vector of feature values defining the intervals
        self.feature_values = None

        # total number of features
        self.nb_features = None

        # frequency of each crossed modality present in the population
        self.crossed_modalities_frequencies = None

        # crossed modalities matrix
        self.crossed_modalities_matrix = None

        # optimization constraints
        self.constraints = None

        # optimization result
        self.optim_result = None

        self.log("Initialisation of enrichment algorithm data", lg.INFO)

        self._init_distributions(distributions, attribute_selection)
        self._init_population(population)

    def _init_distributions(self, distributions, attribute_selection):
        """
        Validate and filter the input distributions.

        When done, set the *distributions* field.

        :param distributions: input distributions DataFrame
        :param attribute_selection: distribution attributes selection
        """

        self.log("Setup distributions data")

        # validate distributions format and contents
        functions.validate_distributions(distributions, attribute_selection)

        distributions = distributions.copy()

        # filter distributions and infer modalities
        self.distributions, self.modalities = functions.filter_distributions_and_infer_modalities(
            distributions, attribute_selection
        )

        # check that there are modalities at the end
        assert (
            len(self.modalities.keys()) > 0
        ), "No attributes found in distributions for enriching population"

    def _init_population(self, population):
        """
        Validate and filter the input population.

        When done, set the *population* field.

        :param population: input population DataFrame
        """
        self.log("Setup population data")

        functions.validate_population(population, self.modalities)

        self.population = population.copy()

        # TODO ? remove distributions unused by population

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
        self.feature_values = functions.compute_feature_values(
            self.distributions, self.parameters["relative_maximum"], self.parameters["delta_min"]
        )
        self.nb_features = len(self.feature_values)
        self.log("Number of feature values: {}".format(self.nb_features))

        # compute matrix of constraints
        self.log("Computing optimization constraints", lg.INFO)
        self._compute_constraints()

        # run resolution
        self.log("Starting optimization by entropy maximisation", lg.INFO)
        self.optim_result = self._run_optimization()

        return self.optim_result

    def _run_optimization(self) -> pd.DataFrame:
        """
        Run optimization model on each feature value.

        The resulting probabilities are the :math:`P(M_{k} \mid f < F_{i})`.

        :return: DataFrame containing the result probabilities
        """

        self.crossed_modalities_matrix = self._compute_crossed_modalities_matrix()
        nb_lines, nb_cols = self.crossed_modalities_matrix.shape

        res = pd.DataFrame()
        lambda_ = np.zeros(nb_lines - 1)

        # loop on features
        for i in range(self.nb_features):
            # run optimization model on the current feature
            self.log("Running optimization algorithm on feature " + str(i))

            q = self.crossed_modalities_frequencies["probability"].values.copy()
            ######### Change to start a prior uniform  tested but it does not change a lot the results
            # dimension = len(q)  # Dimension du vecteur
            # value = 1 / dimension  # Valeur pour toutes les composantes
            # q = np.full(dimension, value)  # Création du vecteur

            k = [1]
            for attribute in self.modalities:
                for modality in self.modalities[attribute][:-1]:
                    k.append(self.constraints[attribute][modality][i])
            k = np.array([k])

            res.loc[:, i], lambda_ = minxent_gradient(
                q=q, matrix=self.crossed_modalities_matrix, eta=k, lambda_=lambda_, maxiter=1000
            )

        return res

    def _compute_crossed_modalities_matrix(self):
        """
        Compute crossed modalities matrix for the present modalities.

        A reducted samplespace is evaluated from the crossed modalities present in the
        population. Functions describing each modality are then applied to elements of
        this samplespace.

        For each modality m and sample c, M(m, c) is 1 if c has modality m, 0 otherwise.

        :return: crossed_modalities_matrix describing crossed modalities
        """

        # samplespace is the set of all possible combinations
        attributes = functions.get_attributes(self.modalities)

        # get samplespace
        samplespace_reducted = self.crossed_modalities_frequencies[attributes].to_dict(
            orient="records"
        )

        features = []

        # base feature is x in samplespace
        def f0(x):
            return x in samplespace_reducted

        features.append(f0)

        # add a feature for all modalities except one for all variables
        for attribute in attributes:
            for modality in self.modalities[attribute][:-1]:
                features.append(functions.modality_feature(attribute, modality))

        nb_lines = len(features)
        nb_cols = len(samplespace_reducted)

        crossed_modalities_matrix = np.zeros((nb_lines, nb_cols))
        for i, f_i in enumerate(features):
            for j in range(nb_cols):
                f_i_x = f_i(samplespace_reducted[j])
                if f_i_x != 0:
                    crossed_modalities_matrix[i, j] = f_i_x

        return crossed_modalities_matrix

    def get_prior_prob(self):
        def function_prior_prob(x_array):
            return self.crossed_modalities_frequencies["probability"].apply(math.log)

        return function_prior_prob

    def _compute_constraints(self):
        """
        For each modality of each attribute, compute the probability of belonging to each feature interval.

        .. math::
            P(Modality \mid f < F_{i}) = P(f < F_{i} \mid Modality) \\cdot \\frac{P(Modality)}{P(f < F_{i})}
        """

        # compute constraints on each modality
        attributes = functions.get_attributes(self.modalities)

        ech = {}
        constraints = {}
        for attribute in attributes:
            ech[attribute] = {}
            # get attribute frequency
            attribute_freq = self.crossed_modalities_frequencies.groupby(
                [attribute], as_index=False
            )["probability"].sum()
            for modality in self.modalities[attribute]:
                self.log("Computing constraints for modality: {} = {}".format(attribute, modality))

                # compute probability of each feature interval when being in a modality
                ech[attribute][modality] = self._compute_feature_prob(attribute, modality)

                # multiply frequencies by each element of ech_compo
                value = attribute_freq[attribute_freq[attribute].isin([modality])]
                if len(value) > 0:
                    probability = value["probability"]
                else:
                    probability = 0
                df = ech[attribute][modality]
                # prob(feature | modality) * frequency // ech is modified inplace here
                df["prob"] = df["prob"] * float(probability)

            ech_list = []
            for modality in ech[attribute]:
                ech_list.append(ech[attribute][modality])
            C = pd.concat(
                ech_list,
                axis=1,
            )
            # Somme P(feature & modality) sur les modality = P(feature)
            C = C.iloc[:, 1::2]
            C.columns = list(range(0, len(ech[attribute])))
            C["Proba"] = C.sum(axis=1)
            p = C[["Proba"]]

            # constraint
            constraints[attribute] = {}
            for modality in ech[attribute]:
                constraints[attribute][modality] = ech[attribute][modality]["prob"] / p["Proba"]

        self.constraints = constraints

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
            float(decile_tmp["D1"]),
            float(decile_tmp["D2"]),
            float(decile_tmp["D3"]),
            float(decile_tmp["D4"]),
            float(decile_tmp["D5"]),
            float(decile_tmp["D6"]),
            float(decile_tmp["D7"]),
            float(decile_tmp["D8"]),
            float(decile_tmp["D9"]),
            float(decile_tmp["D9"]) * self.parameters["relative_maximum"],
        ]

        prob_df = functions.compute_features_prob(self.feature_values, total_population_decile_tmp)

        return prob_df

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
            last = res.iloc[i, -1]
            res.iloc[i, :] = res.iloc[i, :] / last
        pd.set_option("display.max_rows", 500)
        pd.set_option("display.max_columns", 500)
        pd.set_option("display.width", 1000)

        cumulated_results = res.to_numpy()

        matrix = []

        for l in range(len(res)):
            last = cumulated_results[l, 0]
            row = [last]
            for i in range(1, self.nb_features):
                prob = cumulated_results[l, i] - last
                if prob <= 0:
                    row.append(np.nan)
                else:
                    row.append(prob)

                    last = cumulated_results[l, i]

            matrix.append(row)

        probs = pd.DataFrame(matrix, columns=res.columns)

        return probs

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

        The resulting DataFrame contains :math:`P(f < F_{i})` for i in N

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

    # utils methods

    def log(message: str, level: int = lg.DEBUG):
        """
        Log a message using the package logger.

        See logging library.

        :param message: message to be logged
        :param level: logging level
        """

        utils.log(message, level)

    log = staticmethod(log)
