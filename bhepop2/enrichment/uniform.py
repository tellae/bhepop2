from .base import SyntheticPopulationEnrichment, QuantitativeAttributes

import numpy as np


class QuantitativeUniform(SyntheticPopulationEnrichment, QuantitativeAttributes):

    def __init__(
        self,
        population,
        distributions,
        feature_name: str = None,
        seed=None,
        abs_minimum: int = 0,
        relative_maximum: float = 1.5,
            ):

        self.distributions = distributions

        QuantitativeAttributes.__init__(
            self,
            abs_minimum=abs_minimum,
            relative_maximum=relative_maximum
        )

        SyntheticPopulationEnrichment.__init__(
            self,
            population,
            feature_name=feature_name,
            seed=seed,
        )

    def _assign_features(self):

        rng = np.random.RandomState(self.seed)

        centiles = np.array([self._abs_minimum] + self.feature_values)

        pop_size = len(self.population)

        indices = rng.randint(10, size=pop_size)
        lower_bounds, upper_bounds = centiles[indices], centiles[indices + 1]

        incomes = lower_bounds + rng.random_sample(size=pop_size) * (upper_bounds - lower_bounds)

        self.population[self.feature_name] = incomes
        self.population[self.feature_name] = round(self.population[self.feature_name])

        return self.population

    def _validate_and_process_inputs(self):
        pass

    def _evaluate_feature_values(self):
        values = list(self.distributions[["D1", "D2", "D3", "D4", "D5", "D6", "D7", "D8", "D9"]].iloc[0].values)
        values.append(np.max(values)*self._relative_maximum)

        return values
