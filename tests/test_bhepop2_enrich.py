from bhepop2.gradient_enrich import MaxEntropyEnrichment_gradient

import numpy as np
import pandas as pd


def test_bhepop2_enrich(
    synthetic_population_nantes,
    filosofi_distributions_nantes,
    test_modalities,
    test_parameters,
    test_seed,
):
    enrich_class = MaxEntropyEnrichment_gradient(
        synthetic_population_nantes,
        filosofi_distributions_nantes,
        list(test_modalities.keys()),
        parameters=test_parameters,
        seed=test_seed,
    )

    enrich_class.optimise()

    pop = enrich_class.assign_feature_value_to_pop()

    # pop.to_csv("tests/nantes_enriched_gradient.csv", index=False)

    expected_enriched_pop = pd.read_csv("tests/nantes_enriched_gradient.csv")

    assert np.all((pop == expected_enriched_pop).to_numpy())
