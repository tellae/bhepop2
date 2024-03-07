from bhepop2.enrichment.bhepop2 import Bhepop2Enrichment
from bhepop2.sources.marginal_distributions import QuantitativeMarginalDistributions

import numpy as np


def test_bhepop2_enrich(
    synthetic_population_nantes,
    filosofi_distributions_nantes,
    test_modalities,
    test_parameters,
    test_seed,
    expected_enriched_population_nantes,
):
    synthetic_population_nantes.drop("sex", axis=1, inplace=True)

    distributions = QuantitativeMarginalDistributions(
        filosofi_distributions_nantes,
        attribute_selection=list(test_modalities.keys()),
        **test_parameters
    )

    enrich_class = Bhepop2Enrichment(
        synthetic_population_nantes,
        distributions,
        seed=test_seed,
    )

    pop = enrich_class.assign_feature_values()

    pop["feature"] = pop["feature"].astype(int)

    # pop.to_csv("tests/data/nantes_enriched.csv", index=False)

    assert np.all((pop == expected_enriched_population_nantes).to_numpy())
