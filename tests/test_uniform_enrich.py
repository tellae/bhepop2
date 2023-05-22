from bhepop2.uniform_enrich import uniform_enrich


def test_uniform_enrich(synthetic_population_nantes, filosofi_distributions_nantes, test_seed):
    filosofi_distributions_nantes = filosofi_distributions_nantes[
        filosofi_distributions_nantes["modality"] == "all"
    ]

    pop = uniform_enrich(
        synthetic_population_nantes, filosofi_distributions_nantes, 0, 1.2, test_seed
    )
