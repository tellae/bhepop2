import numpy as np


# TODO : refactor this function in a class similar to QuantitativeEnrichment (started below)
def uniform_enrich(population, distributions, abs_minimum, relative_maximum, seed):
    """

    :param population: synthetic population
    :param distributions: decile distribution (modality 'all')
    :param abs_minimum: see QuantitativeEnrichment parameters
    :param relative_maximum: see QuantitativeEnrichment parameters
    :param seed: random seed

    :return: enriched population
    """

    rng = np.random.RandomState(seed)

    centiles = list(
        distributions[["D1", "D2", "D3", "D4", "D5", "D6", "D7", "D8", "D9"]].iloc[0].values
    )

    centiles = np.array([abs_minimum] + centiles + [np.max(centiles) * relative_maximum])

    pop_size = len(population)

    indices = rng.randint(10, size=pop_size)
    lower_bounds, upper_bounds = centiles[indices], centiles[indices + 1]

    incomes = lower_bounds + rng.random_sample(size=pop_size) * (upper_bounds - lower_bounds)

    population["feature"] = incomes
    population["feature"] = round(population["feature"])

    return population


# class UniformEnrich:
#
#     def __init__(self, population, distributions, abs_minimum, relative_maximum, seed):
#
#         # random generator
#         self.rng = np.random.RandomState(seed)
#
#         # original population to be enriched
#         self.population = None
#
#         # distributions of the feature values by modality
#         self.distributions = None
#
#         self.abs_minimum = abs_minimum
#         self.relative_maximum = relative_maximum
#
#         # algorithm data
#
#         # vector of feature values defining the intervals
#         self.feature_values = None
#
#         # total number of features
#         self.nb_features = None
#
#         self.result_probs = None
#
#         self.log("Initialisation of enrichment algorithm data", lg.INFO)
#
#         self.log("Setup distributions data")
#         self._init_distributions(distributions)
#
#         self.log("Setup population data")
#         self._init_population(population)
#
#     def _init_distributions(self, distributions):
#         self.distributions = distributions.copy()
#
#         # TODO : assert only one line with attribute all
#
#     def _init_population(self, population):
#         self.population = population.copy()
#
#     def evaluate_probs(self):
#
#         centiles = list(self.distributions[
#                             ["D1", "D2", "D3", "D4", "D5", "D6", "D7", "D8", "D9"]].iloc[0].values)
#
#         centiles = np.array([self.abs_minimum] + centiles + [np.max(centiles) * self.relative_maximum])
#
#         self.feature_values = centiles
#
#     def draw_feature_values(self):
#
#         pop_size = len(self.population)
#
#         indices = self.rng.randint(10, size=pop_size)
#         lower_bounds, upper_bounds = self.feature_values[indices], self.feature_values[indices + 1]
#
#         incomes = lower_bounds + self.rng.random_sample(size=pop_size) * (upper_bounds - lower_bounds)
#
#         self.population["feature"] = incomes
#         self.population["feature"] = round(self.population["feature"])
#
#         return self.population
#
#     def log(message: str, level: int = lg.DEBUG):
#         """
#         Log a message using the package logger.
#
#         See logging library.
#
#         :param message: message to be logged
#         :param level: logging level
#         """
#
#         utils.log(message, level)
#
#     log = staticmethod(log)
