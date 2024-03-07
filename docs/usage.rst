.. _usage:


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

   * - 15 000
     - ...
     - 45 000
     - age
     - 0_29
   * - ...
     - ...
     - ...
     - ...
     - ...
   * - 20 000
     - ...
     - 65 000
     - age
     - 75_or_more

Population
**********

This is the **synthetic population to be enriched** using Bhepop2.
Sources for synthetic populations can be *Eqasim* population/households, but any list of individuals is virtually
a synthetic population.

Populations are represented by a **DataFrame containing one entry per population individual**.
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


Enrich a population using Bhepop2
#################################

Let's continue with the tables of the previous examples. We want to enrich our population with
income data.

.. code-block:: python

    import pandas as pd

    # population with ownership and age attributes (see previous example)
    synth_pop = pd.read_csv("./synthetic_population.csv")

Enrichment method
*****************

The first step is to choose a method to enrich our synthetic population.
Currently, the following enrichment classes are available:

.. autosummary::
    :nosignatures:

    ~bhepop2.enrichment.uniform.SimpleUniformEnrichment
    ~bhepop2.enrichment.bhepop2.Bhepop2Enrichment

Following our example, we choose the :class:`~bhepop2.enrichment.bhepop2.Bhepop2Enrichment` class.

Enrichment source
*****************

From the class documentation, we see that it expects an instance of either
:class:`~bhepop2.sources.marginal_distributions.QualitativeMarginalDistributions`
or :class:`~bhepop2.sources.marginal_distributions.QuantitativeMarginalDistributions` as a source. Here, income data
comes in quantitative distributions, so we use :class:`~bhepop2.sources.marginal_distributions.QuantitativeMarginalDistributions`.

.. code-block:: python

    from bhepop2.source.marginal_distributions import QuantitativeMarginalDistributions

    # marginal distributions for ownership and age attributes (see previous example)
    income_distributions = pd.read_csv("./income_distributions.csv")

    # create an instance of QuantitativeMarginalDistributions
    income_source = QuantitativeMarginalDistributions(
        marginal_distributions,
        attribute_selection=["age", "ownership"],  # distribution attributes used for enrichment
        abs_minimum=0,  # absolute value used as a minimum for all distributions
        relative_maximum=1.5,  # relative value multiplied to each distribution last value to evaluate a maximum
        name="Example source",  # name of the source, used in displays
    )

Population enrichment
*********************

Then we initialise our enrichment class instance with the population and source, and call the feature assignment method.

.. code-block:: python

    from bhepop2.enrichment.bhepop2 import Bhepop2Enrichment

    enrich_class = Bhepop2Enrichment(
        synth_pop,  # synthetic population to be enriched
        income_source,  # enrichment source
        feature_name="income",  # column added to the population DataFrame
        seed=42,  # random seed, for reproducing results
    )

    enriched_population = enrich_class.assign_features()

The resulting population DataFrame presents a new **income** column with values evaluated using the Bhepop2 methodology.

.. list-table:: Example of enriched population with new income information
   :widths: 30 10 25 25 25
   :header-rows: 1

   * - id
     - ...
     - ownership
     - age
     - income
   * - u-1.1
     - ...
     - Owner
     - 60_74
     - 50 000
   * - u-1.2
     - ...
     - Tenant
     - 0_29
     - 22 000
   * - u-1.3
     - ...
     - Tenant
     - 40_49
     - 40 000


Example notebooks
*****************

For more detailed examples, see the `examples folder <https://github.com/tellae/bhepop2/tree/main/examples>`_ in the GitHub repository.