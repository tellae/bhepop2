from bhepop2.enrichment import QuantitativeUniform


def test_uniform_enrich(synthetic_population_nantes, filosofi_distributions_nantes, test_seed):
    filosofi_distributions_nantes = filosofi_distributions_nantes[
        filosofi_distributions_nantes["modality"] == "all"
    ]

    enrich_class = QuantitativeUniform(synthetic_population_nantes, filosofi_distributions_nantes, abs_minimum=0, relative_maximum=1.2)

    pop = enrich_class.assign_features()
