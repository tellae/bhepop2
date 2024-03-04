.. _inputs:


Usage
_____

Input data
##########

The two main inputs of the enrichment algorithms are the population to be enriched and the aggregated data.

Enrichment source
*****************

The enrichment source contain the **information used by the algorithm to enrich the population**.
Source data for such data can be *INSEE* databases (*Filosofi*, ..) or other aggregated data sources.

The source data formatting is not fixed, as it depends on the

.. note::

    The source data format is not fixed. It depends of the method used for population enrichment.

Example
-------

When enriching populations with :class:`~bhepop2.enrichment.bhepop2.Bhepop2Enrichment`, we use marginal distributions,
meaning distributions that describe a subset of the population with a specific attribute value.

For instance, we may use distributions that describe a population depending on its age, and other distributions may
use the individuals ownership situation.

They are represented by a DataFrame containing one entry per distribution, for each attribute and its values (called modalities).


.. list-table:: Example of a table containing marginal distributions for attributes **ownership** and **age**
   :widths: 25 10 25 40 40
   :header-rows: 1

   * - D1
     - ...
     - D9
     - attribute
     - modality
   * - 16 542
     - ...
     - 50 060
     - ownership
     - Owner
   * - 8 764
     - ...
     - 29 860
     - ownership
     - Tenant

Population
**********

This is the **synthetic population to be enriched** using Bhepop2.
Sources for synthetic populations can be *Eqasim* population/households, but any list of individuals is virtually
a synthetic population.

Synthetic populations are represented by a **DataFrame containing one entry per population individual**.
Some other characteristics may be required on the population depending on the enrichment methodology and data source.

Example
-------

Following our previous example, we have marginal distributions depending on age and ownership.

In order to match the distributions with the population, the DataFrame must have columns describing these attributes.

.. list-table:: Example of population table with **ownership** and **age** columns (the **id** column is not mandatory)
   :widths: 30 10 25 25
   :header-rows: 1

   * - id
     - ...
     - ownership
     - age
   * - u-1.1
     - ...
     - Owner
     - 60_74
   * - u-1.2
     - ...
     - Tenant
     - 0_29
   * - u-1.3
     - ...
     - Tenant
     - 40_49


Using the Bhepop2 library
#########################

Enrichment method
*****************

Enrichment classes are used

.. autosummary::
    :nosignatures:

    ~bhepop2.enrichment.uniform.SimpleUniformEnrichment
    ~bhepop2.enrichment.bhepop2.Bhepop2Enrichment

Enrichment source data
**********************

Encapsulates aggregated data.

.. autosummary::
    :nosignatures:

    ~bhepop2.sources.global_distribution.QuantitativeGlobalDistribution
    ~bhepop2.sources.marginal_distributions.QualitativeMarginalDistributions
    ~bhepop2.sources.marginal_distributions.QuantitativeMarginalDistributions




