from bhepop2.quantitative_enrichment import QuantitativeEnrichment
from bhepop2.utils import *


def test_add_defaults_and_validate_against_schema(test_parameters):
    """
    Test the json schema validation.
    """

    add_defaults_and_validate_against_schema(test_parameters, QuantitativeEnrichment.parameters_schema)
