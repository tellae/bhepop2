from bhepop2.analysis import *


def test_analyse_enriched_population(
    filosofi_distributions_nantes, expected_enriched_population_nantes, test_modalities, tmp_dir
):
    populations = {"enriched": expected_enriched_population_nantes}

    analyse_enriched_populations(
        populations,
        filosofi_distributions_nantes,
        "Filosofi",
        test_modalities,
        tmp_dir,
        plots=False,
    )


def test_compute_distribution(expected_enriched_population_nantes):
    """
    Test that computed distribution is exactly the same as expected
    """
    expected = [
        {"feature": 9820.0, "decile": "D1"},
        {"feature": 13259.0, "decile": "D2"},
        {"feature": 16095.8, "decile": "D3"},
        {"feature": 18634.4, "decile": "D4"},
        {"feature": 21382.0, "decile": "D5"},
        {"feature": 24390.6, "decile": "D6"},
        {"feature": 28065.0, "decile": "D7"},
        {"feature": 33036.8, "decile": "D8"},
        {"feature": 42485.0, "decile": "D9"},
    ]

    assert (
        compute_distribution(expected_enriched_population_nantes).round(1).to_dict(orient="records")
        == expected
    )
