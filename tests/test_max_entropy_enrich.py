from bhepop2.max_entropy_enrich import MaxEntropyEnrichment

import numpy as np


def test_max_entropy_enrich(
    synthetic_population_nantes,
    filosofi_distributions_nantes,
    test_modalities,
    test_parameters,
    test_seed,
    expected_enriched_population_nantes
):
    enrich_class = MaxEntropyEnrichment(
        synthetic_population_nantes,
        filosofi_distributions_nantes,
        list(test_modalities.keys()),
        parameters=test_parameters,
        seed=test_seed,
    )

    enrich_class.optimise()

    pop = enrich_class.assign_feature_value_to_pop()

    assert np.all((pop == expected_enriched_population_nantes).to_numpy())
