from bhepop2.bhepop2_enrichment import Bhepop2Enrichment

import numpy as np
import pytest


@pytest.skip  # runs locally but not on gh actions
def test_bhepop2_enrich(
    synthetic_population_nantes,
    filosofi_distributions_nantes,
    test_modalities,
    test_parameters,
    test_seed,
    expected_enriched_population_nantes,
):
    enrich_class = Bhepop2Enrichment(
        synthetic_population_nantes,
        filosofi_distributions_nantes,
        list(test_modalities.keys()),
        parameters=test_parameters,
        seed=test_seed,
    )

    enrich_class.optimise()

    pop = enrich_class.assign_feature_value_to_pop()

    # pop.to_csv("tests/data/nantes_enriched.csv", index=False)

    assert np.all((pop == expected_enriched_population_nantes).to_numpy())
