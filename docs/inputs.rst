.. _inputs:

######
Inputs
######

The two main inputs of the enrichment algorithm are the population to be enriched and the aggregated data.

They are linked by a set of **attributes** and their possible values (**modalities**). For instance:

- ``Ownership``, with values in [``Owner``, ``Tenant``]
- ``Age``, with values in [``0_29``, ..., ``60_74``, ``75_or_more``]

**********
Population
**********

This is the **synthetic population to be enriched** using Bhepop2.

It is represented by a DataFrame containing one entry per population individual.
Columns of the same name than the attributes are expected. Entries contain a value (modality) of each attribute.

.. list-table:: Example of population table
   :widths: 25 25 25
   :header-rows: 1

   * - ...
     - ownership
     - age
   * - ...
     - Owner
     - 60_74
   * - ...
     - Tenant
     - 0_29
   * - ...
     - Tenant
     - 40_49

Source data for the formatted population can be *Eqasim* population/households or any source of synthetic population.

*************
Distributions
*************

Distributions contain the **information used by the algorithm to enrich the population**.

They are represented by a DataFrame containing one entry per distribution. The expected columns are the following:


- ``commune_id`` : the commune identifier to which the distribution is associated
- ``D1``; ``D2``; ...; ``D9`` : deciles of the distribution
- ``attribute`` : name of the attribute associated to this distribution
- ``modality`` : name of the modality associated to this distribution


.. list-table:: Example of distribution table
   :widths: 25 25 10 25 40 40
   :header-rows: 1

   * - commune_id
     - D1
     - ...
     - D9
     - attribute
     - modality
   * - 44109
     - 16 542
     - ...
     - 50 060
     - ownership
     - Owner
   * - 44109
     - 8 764
     - ...
     - 29 860
     - ownership
     - Tenant

.. note::

    The values of the ``attribute`` column define the columns that must be present in the population data.

    The values of the ``modality`` column define the possible values that can be taken by the corresponding attribute in
    the population data.

Source data for the formatted distributions can be *INSEE* databases (*Filosofi*, ..) or other aggregated data sources.
