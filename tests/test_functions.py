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

def test_infer_modalities_from_distributions():
    filosofi = get_filosofi_distributions()
    filosofi = filosofi.query(f"commune_id == '{CODE_INSEE}'")

    modalities = infer_modalities_from_distributions(filosofi)

    assert isinstance(modalities, dict)
    assert "all" not in modalities

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

def test_interpolate_feature_prob():
    """
    Test that the feature probability interpolation is a float and has a coherent value.
    """

    distribution = [9794, 12961, 14914, 16865, 18687, 20763, 23357, 27069, 33514, 50271]
    feature_value = 16000
    interpolation = interpolate_feature_prob(feature_value, distribution)

    assert isinstance(interpolation, float)
    assert 0.3 <= interpolation <= 0.4