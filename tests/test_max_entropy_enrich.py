from src.max_entropy_enrich import MaxEntropyEnrichment
from tests.conftest import *
import numpy as np

parameters = {
    "abs_minimum": 0,
    "relative_maximum": 1.5,
    "maxentropy_algorithm": "Nelder-Mead",
    "maxentropy_verbose": 0
}

def test_max_entropy_enrich():
    synth_pop = get_synth_pop_nantes()

    filosofi = get_filosofi_distributions()

    enrich_class = MaxEntropyEnrichment(synth_pop, filosofi, CODE_INSEE, list(MODALITIES.keys()), parameters=parameters)

    res = enrich_class.main()

    expected = pd.read_csv("../tests/nantes_result.csv")
    expected.columns = [int(x) for x in expected.columns]

    assert np.all(np.isclose(expected.to_numpy(), res.to_numpy()))

    # constraints_old = functions.create_constraints(MODALITIES, enrich_class.distributions, enrich_class.feature_values, enrich_class.crossed_modalities_frequencies)
    # for attribute in MODALITIES:
    #     for modality in MODALITIES[attribute]:
    #         assert (constraints[attribute][modality] == constraints_old[attribute][modality]).all()