"""
This is the main bhepop2 package.
"""

#: bhepop2 version
__version__ = "1.0.0"

# from bhepop2.bhepop2_enrichment import QuantitativeEnrichment
#
#
# # enrichment functions (avoids direct use of classes)
#
# def quantitative_enrichment(population, distributions, attribute_selection=None, parameters=None, seed=None):
#     """
#     Enrich the given synthetic population with a quantitative feature using marginal distributions.
#
#     :param population:
#     :param distributions:
#     :param attribute_selection:
#     :param parameters:
#     :param seed:
#
#     :return: enriched population
#     """
#     # create enrichment class instance
#     enrich_class = QuantitativeEnrichment(
#         population,
#         distributions,
#         attribute_selection=attribute_selection,
#         parameters=parameters,
#         seed=seed,
#     )
#
#     # run entropy maximisation
#     enrich_class.optimise()
#
#     # return enriched population
#     return enrich_class.assign_feature_value_to_pop()
#
