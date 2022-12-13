from tests.conftest import *

from src.functions2 import *
import numpy as np


def test_compute_crossed_modalities_frequencies():
    """
    Test the crossed modalities frequencies have the correct columns and sum to 1.
    """
    synth_pop = get_synth_pop_nantes()
    freq_df = compute_crossed_modalities_frequencies(synth_pop, MODALITIES)

    assert list(freq_df.columns) == get_attributes(MODALITIES) + ["probability"]
    assert np.isclose(freq_df["probability"].sum(), 1)

def test_compute_feature_values():
    """
    Test that the feature values list have the correct length and is sorted.
    """

    filosofi = get_filosofi_distributions()
    filosofi = filosofi.query(f"commune_id == '{CODE_INSEE}'")
    filosofi = filosofi[filosofi["attribute"].isin(get_attributes(MODALITIES))]
    feature_values = compute_feature_values(filosofi, 0, 1.5)

    assert len(feature_values) == 190
    assert feature_values == sorted(feature_values)
