from bhepop2.enrichment import SimpleUniformEnrichment
from bhepop2.sources.global_distribution import QuantitativeGlobalDistribution


def test_uniform_enrich(synthetic_population_nantes, filosofi_distributions_nantes, test_seed):
    filosofi_distributions_nantes = filosofi_distributions_nantes[
        filosofi_distributions_nantes["modality"] == "all"
    ]

    global_distribution = QuantitativeGlobalDistribution(filosofi_distributions_nantes, abs_minimum=0, relative_maximum=1.2)

    enrich_class = SimpleUniformEnrichment(synthetic_population_nantes, global_distribution)

    pop = enrich_class.assign_features()
