from bhepop2.bhepop2_enrichment import Bhepop2Enrichment
from bhepop2.utils import *


def test_add_defaults_and_validate_against_schema(test_parameters):
    """
    Test the json schema validation.
    """

    add_defaults_and_validate_against_schema(
        test_parameters, Bhepop2Enrichment.parameters_schema
    )
