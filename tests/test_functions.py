from tests.conftest import *
from bhepop2.functions import *
import numpy as np

import pytest


def test_get_attributes():
    """
    Test that the returned attributes are correct.
    """

    assert get_attributes(MODALITIES) == ["ownership", "age", "size", "family_comp"]


def test_modality_feature():
    """
    Test that the result is a function with expected behaviour.
    """

    attribute = "my_attribute"
    modality_0 = "modality_0"
    modality_1 = "modality_1"

    feature = modality_feature(attribute, modality_0)

    assert callable(feature)
    assert feature({attribute: modality_0})
    assert not feature({attribute: modality_1})


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


@pytest.mark.parametrize(
    "delta_min, expected_length",
    [
        (None, 190),
        (1000, 41),
    ],
)
def test_compute_feature_values(delta_min, expected_length):
    """
    Test that the feature values list have the correct length and is sorted.
    """

    filosofi = get_filosofi_distributions()
    filosofi = filosofi.query(f"commune_id == '{CODE_INSEE}'")
    filosofi = filosofi[filosofi["attribute"].isin(get_attributes(MODALITIES))]
    feature_values = compute_feature_values(filosofi, 1.5, delta_min)

    assert len(feature_values) == expected_length
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
