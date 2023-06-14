

from bhepop2.gradient_enrich import MaxEntropyEnrichment_gradient
from bhepop2.max_entropy_enrich import MaxEntropyEnrichment
from bhepop2.tools import read_filosofi, filosofi_attributes
from bhepop2.analysis import analyse_enriched_populations
import bhepop2.utils as utils

import pandas as pd
import os

ANALYSIS_OUTPUT_FOLDER = "plots/test_gradient0/"
if not os.path.exists(ANALYSIS_OUTPUT_FOLDER):
    os.mkdir(ANALYSIS_OUTPUT_FOLDER)

MODALITIES = {
    "ownership": ["Owner", "Tenant"],
    "age": ["0_29", "30_39", "40_49", "50_59", "60_74", "75_or_more"],
    "size": ["1_pers", "2_pers", "3_pers", "4_pers", "5_pers_or_more"],
    "family_comp": [
        "Single_man",
        "Single_wom",
        "Couple_without_child",
        "Couple_with_child",
        "Single_parent",
        "complex_hh",
    ],
}

PARAMETERS = {
    "abs_minimum": 0,
    "relative_maximum": 1.2,
    "delta_min": 1000,
}

utils.logger_level = 10  # set logger level to INFO

# evaluate gradient enriched population

# get synthetic population
synth_pop = pd.read_csv("tests/data/nantes_synth_pop.csv", sep=";")

# get filosofi distributions
filosofi = read_filosofi("tests/data/FILO_DISP_COM.xls", "15", filosofi_attributes, ["44109"])

# create enrich class and generate enriched population
enrich_class = MaxEntropyEnrichment_gradient(
        synth_pop,
        filosofi,
        list(MODALITIES.keys()),
        parameters=PARAMETERS,
        seed=42,
    )
enrich_class.optimise()

gradient_enriched = enrich_class.assign_feature_value_to_pop()
utils.log("End of gradient enrichment", 20)

# get reference enriched population (with previous enrich class MaxEntropyEnrichment)
# reference_enriched = pd.read_csv("tests/data/nantes_enriched.csv")

enrich_maxentropy = MaxEntropyEnrichment(
    synth_pop,
    filosofi,
    list(MODALITIES.keys()),
    parameters=PARAMETERS,
    seed=42
)
enrich_maxentropy.optimise()

reference_enriched = enrich_maxentropy.assign_feature_value_to_pop()

# compare enriched populations using plots and error table

utils.log("Generating compared analysis between reference and gradient enrichment", 20)

populations = {
    "reference": reference_enriched,
    "gradient": gradient_enriched
}

analyse_enriched_populations(
    populations,
    filosofi,
    "Filosofi",
    MODALITIES,
    ANALYSIS_OUTPUT_FOLDER,
)
