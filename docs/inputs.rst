.. _inputs:

######
Inputs
######

**********
Population
**********

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

Source data for the formatted distributions can be *INSEE* databases (*Filosofi*, ..) or other demographic data sources.
