from src.max_entropy_enrich import MaxEntropyEnrichment
from src.utils import *
from tests.conftest import *

def test_add_defaults_and_validate_against_schema():
    """
    Test the json schema validation.
    """

    instance = parameters
    add_defaults_and_validate_against_schema(instance, MaxEntropyEnrichment.parameters_schema)

