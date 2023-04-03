from bhepop2.max_entropy_enrich import MaxEntropyEnrichment
from bhepop2.utils import *


def test_add_defaults_and_validate_against_schema(test_parameters):
    """
    Test the json schema validation.
    """

    add_defaults_and_validate_against_schema(test_parameters, MaxEntropyEnrichment.parameters_schema)
