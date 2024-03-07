"""
Implementation of the Bhepop2 methodology.

Bhepop2 stands for Bayesian Heuristic to Enrich POPulation by EntroPy OPtimization. It is theoretically described, justified and discussed in

* Boyam Fabrice Yaméogo, Pierre-Olivier Vandanjon, Pierre Hankach, Pascal Gastineau. `Methodology for Adding a Variable to a Synthetic Population from Aggregate Data: Example of the Income Variable <https://hal.archives-ouvertes.fr/hal-03282111>`_. 2021. ⟨hal-03282111⟩. Paper in review.

* Boyam Fabrice Yaméogo, `Méthodologie de calibration d’un modèle multimodal des déplacements pour l’évaluation des externalités environnementales à partir de données ouvertes (open data) : le cas de l’aire urbaine de Nantes [Thèse] <https://www.theses.fr/2021NANT4085>`_, 2021

Diagram representation of the algorithm:

.. mermaid::

    graph TD
         Pop[("Population <br /> with Attributes <br /> A,B")] -->Frequencies("Cross Modalities Frequencies <br /> F(A #8745; B)")
         Distribution("Distribution (Deciles) <br /> P(I #8712; [Id,Id+1] | A) <br /> P(I #8712; [Ik,Ik+1] | B)") -->
         Interpolation("Sorted deciles & Interpolation <br /> P(I | A) <br /> P(I | B)")
         Entropy(" Cross Entropy Optimizations <br /> P( (A #8745; B) | I ) <br /> under the constrains <br /> P(I | A) <br /> P(I | B)" )
         Frequencies --> Entropy
         Interpolation --> Entropy
         Entropy --> Bayesian("Bayesian rule <br /> P( I | (A #8745; B)) = P( (A #8745; B) | I ) * P(I)/P(A #8745; B)")
        Bayesian ---> Cleaning("Cleaning <br />inconsistent to <br /> consistent proba ")
        Cleaning -->|Sampling|Sampling[("Population <br /> with Attributes <br /> A,B, I")]
"""

import numpy as np
import pandas as pd
import logging as lg

from .base import SyntheticPopulationEnrichment
from bhepop2.sources.marginal_distributions import MarginalDistributions
from bhepop2 import functions
from bhepop2.optim import minxent_gradient


class Bhepop2Enrichment(SyntheticPopulationEnrichment):
    """
    Implementation of the Bhepop2 methodology as an enrichment class.
    See :mod:`bhepop2.enrichment.bhepop2` module documentation for details about the algorithm.

    **Expected source types**:

    .. autosummary::
        :nosignatures:

        ~bhepop2.sources.marginal_distributions.QualitativeMarginalDistributions
        ~bhepop2.sources.marginal_distributions.QuantitativeMarginalDistributions

    ----

    This class documentation uses the following notations:

    - :math:`M_{k}` : crossed modality k (combination of attribute modalities)

    - :math:`F_{i}` : feature class i

        - For quantitative features, corresponds to a numeric interval.
        - For qualitative features, corresponds to one of the feature values.

    """

    def __init__(
        self,
        population: pd.DataFrame,
        source: MarginalDistributions,
        feature_name=None,
        seed=None,
    ):
        """
        Add some specific fields used during the Bhepop2 computation.

        The Bhepop2 method necessitates an instance of MarginalDistribution
        as a source, ie distributions evaluated per modalities.

        :param population: population to enrich
        :param source: enrichment source
        :param seed: random seed
        """

        # algorithm data

        # frequency of each crossed modality present in the population
        self.crossed_modalities_frequencies = None

        # crossed modalities matrix
        self.crossed_modalities_matrix = None

        # optimization constraints
        self.constraints = None

        # optimization result
        self.optim_result = None

        super().__init__(population, source, feature_name=feature_name, seed=seed)

    @property
    def modalities(self):
        """
        Dict containing list of modalities for each attribute.
        """
        return self.source.modalities

    def _validate_and_process_inputs(self):
        """
        Validate the provided inputs and set the relevant fields.

        Since Bhepop2 uses marginal distributions to enrich the population,
        we ensure that:

        * the selected attributes are present in the population
        * the population attributes take values in the modalities corresponding to this attribute
        """

        assert isinstance(
            self.source, MarginalDistributions
        ), "Bhepop2Enrichment needs a MarginalDistributions source"

        self.log("Setup population data")

        functions.validate_population(self.population, self.modalities)

    def _evaluate_feature_on_population(self):
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
            on=functions.get_attributes(self.source.modalities),
        )

        # associate a feature value to the population individuals
        return merge["index"].apply(lambda x: self._draw_feature_value(res.loc[x,].to_numpy()))

    def _draw_feature_value(self, probs):
        """
        Return a feature value using the given probabilities.

        First draw the feature index.
        Then get a feature value from the distributions.

        :return: feature value to assign to individual
        """

        # get the non-null probs and their indexes
        filtered_probs = []
        feature_indexes = []
        for i in range(len(probs)):
            if pd.isna(probs[i]):
                continue
            filtered_probs.append(probs[i])
            feature_indexes.append(i)

        # draw a feature index using the probs
        feature_index = self.rng.choice(feature_indexes, p=filtered_probs)

        # get feature value from distribution
        value = self._get_value_for_feature(feature_index)

        return value

    def _get_feature_probs(self):
        """
        For each crossed modality, compute the probability of belonging to a feature interval.

        Invert the crossed modality probabilities using Bayes.

        Compute

        .. math::

            P(f \\in F_{i} \\mid M_{k}) = P(M_{k} \\mid f \\in F_{i}) \\cdot \\frac{P(f \\in F_{i})}{P(M_{k})}

        :return: DataFrame
        """

        feature_probs = self.source.compute_feature_prob()

        res = self.optim_result

        nb_columns = len(res.columns)

        for c in res.columns:
            res[c] = res[c].map(lambda x: x[0])  # POV sur conseils de Valentin

        for i in range(nb_columns):
            res[i] = res[i] * feature_probs["prob"][i]

        for i in range(len(res)):
            total = res.iloc[i, :].sum()
            res.iloc[i, :] = res.iloc[i, :] / total

        return res

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
        for i in range(self.source.nb_feature_values):
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
                ech[attribute][modality] = self.source.compute_feature_prob(attribute, modality)

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
            probs = pd.concat(
                ech_list,
                axis=1,
            )
            # Somme P(feature & modality) sur les modality = P(feature)
            probs = probs.iloc[:, 1::2]
            probs.columns = list(range(0, len(ech[attribute])))
            probs["Proba"] = probs.sum(axis=1)
            p = probs[["Proba"]]

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
