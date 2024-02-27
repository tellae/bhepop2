"""
This module contains base code for synthetic population enrichment classes.
"""

from bhepop2.utils import log, lg, add_defaults_and_validate_against_schema

from abc import ABC
import pandas as pd
import random


import numpy as np
import random
from bhepop2 import utils
from bhepop2 import functions
from .optim import minxent_gradient


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

    def __init__(self, population: pd.DataFrame, feature_name: str = "feature", seed=None):

        # random seed (maybe use a random generator instead)
        self.seed = seed
        if seed is not None:
            random.seed(seed)

        # original population DataFrame, to be enriched
        self.population: pd.DataFrame = population

        # name of the added column containing the new feature values
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

    def assign_features(self):
        """
        Assign feature values to the population individuals.

        This method evaluates and adds feature values for each
        population individual.

        The name of the added column is defined by the feature_name class parameter.
        """

        self.enriched_population = self._assign_features()

        return self.enriched_population

    def _assign_features(self):
        raise NotImplementedError

    def analyze_features(self):
        """
        Return or generate an analysis of the added features.

        :return:
        """
        raise NotImplementedError

    def _validate_and_process_inputs(self):
        """

        :return:
        """

    def _evaluate_feature_values(self):
        raise NotImplementedError

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


class Bhepop2Enrichment(SyntheticPopulationEnrichment):

    parameters_schema = {}

    def __init__(
        self,
        population: pd.DataFrame,
        distributions: pd.DataFrame,
        attribute_selection: list = None,
        parameters=None,
        feature_name="feature",
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

        # distributions of the feature values by modality
        self.distributions = distributions

        # distribution attributes used for feature evaluation
        self.attribute_selection = attribute_selection

        # attributes considered for the assignment, with their modalities
        # { attribute: [modalities] }
        self.modalities = None

        # execution parameters
        if parameters is None:
            parameters = dict()
        self.parameters = add_defaults_and_validate_against_schema(
            parameters, self.parameters_schema
        )

        # algorithm data

        # frequency of each crossed modality present in the population
        self.crossed_modalities_frequencies = None

        # crossed modalities matrix
        self.crossed_modalities_matrix = None

        # optimization constraints
        self.constraints = None

        # optimization result
        self.optim_result = None

        super().__init__(population, feature_name=feature_name, seed=seed)

    def _assign_features(self):
        """
        Assign feature values to the population individuals using the algorithm results.

        :return: enriched population DataFrame
        """

        self._optimise()

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
            P(f \\in F_{i} \\mid M_{k}) = P(M_{k} \\mid f \\in F_{i}) \\cdot \\frac{P(f \\in F_{i})}{P(M_{k})}

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

    def _validate_and_process_inputs(self):
        self._init_distributions()
        self._init_population()

    def _init_population(self):
        """
        Validate and filter the input population.

        When done, set the *population* field.
        """
        self.log("Setup population data")

        functions.validate_population(self.population, self.modalities)

        self.population = self.population.copy()

    def _init_distributions(self):
        raise NotImplementedError

    def _optimise(self):
        """
        Run the optimisation algorithm to find the probability distributions that maximise entropy.

        When done, set the *optim_result* attribute.
        """
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

        The resulting probabilities are the :math:`P(M_{k} \\mid f \\in F_{i})`.

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

    def _compute_constraints(self):
        """
        For each modality of each attribute, compute the probability of belonging to each feature interval.

        .. math::
            P(Modality \\mid f \\in F_{i}) = P(f \\in F_{i} \\mid Modality) \\cdot \\frac{P(Modality)}{P(f \\in F_{i})}
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
                    probability = value["probability"].iloc[0]
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

    def _draw_feature(self, res, index):
        raise NotImplementedError


    def _compute_feature_prob(self, attribute, modality):
        raise NotImplementedError

    def _compute_feature_probabilities_from_distributions(self):
        raise NotImplementedError
