from bhepop2.max_entropy_enrich import MaxEntropyEnrichment
from tests.conftest import *
import numpy as np


def test_max_entropy_enrich():
    synth_pop = get_synth_pop_nantes()

    filosofi = get_filosofi_distributions()
    filosofi = filosofi.query(f"commune_id == '{CODE_INSEE}'")

    enrich_class = MaxEntropyEnrichment(
        synth_pop, filosofi, list(MODALITIES.keys()), parameters=parameters, seed=SEED
    )

    enrich_class.optimise()

    pop = enrich_class.assign_feature_value_to_pop()

    pop.to_csv("tests/nantes_enriched.csv", index=False)

    expected_enriched_pop = pd.read_csv("tests/nantes_enriched.csv")

    assert np.all((pop == expected_enriched_pop).to_numpy())
