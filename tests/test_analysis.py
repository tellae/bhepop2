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
        {'feature': 9890.6, 'decile': 'D1'},
        {'feature': 13190.0, 'decile': 'D2'},
        {'feature': 15779.0, 'decile': 'D3'},
        {'feature': 18314.0, 'decile': 'D4'},
        {'feature': 20865.0, 'decile': 'D5'},
        {'feature': 23738.0, 'decile': 'D6'},
        {'feature': 27290.0, 'decile': 'D7'},
        {'feature': 32093.8, 'decile': 'D8'},
        {'feature': 41201.4, 'decile': 'D9'}
    ]

    assert (
        compute_distribution(expected_enriched_population_nantes).round(1).to_dict(orient="records")
        == expected
    )
