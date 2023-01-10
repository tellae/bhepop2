.. _algorithm:

#########
Algorithm
#########

This package is based on a methodology called Bhepop2 (Bayesian Heuristic to Enrich POPulation by EntroPy OPtimization) and theoritically described, justified and discussed in

- `Boyam Fabrice Yaméogo, Pierre-Olivier Vandanjon, Pierre Hankach, Pascal Gastineau. Methodology for Adding a Variable to a Synthetic Population from Aggregate Data: Example of the Income Variable. 2021. ⟨hal-03282111⟩. Paper in review. <https://hal.archives-ouvertes.fr/hal-03282111>`_

- `Boyam Fabrice Yaméogo, Méthodologie de calibration d’un modèle multimodal des déplacements pour l’évaluation des externalités environnementales à partir de données ouvertes (open data) : le cas de l’aire urbaine de Nantes [Thèse], 2021 <https://www.theses.fr/2021NANT4085>`_

.. mermaid::

    graph TD
         Pop[("Population <br /> with Attributes <br /> A,B")] -->Frequencies("Cross Modalities Frequencies <br /> F(A #8745; B)")
         Distribution("Distribution (Deciles) <br /> P(I #8712; [Id,Id+1] | A) <br /> P(I #8712; [Ik,Ik+1] | B)") -->
         Interpolation("Sorted deciles & Interpolation <br /> P(I | A) <br /> P(I | B)")
         Entropy(" Cross Entropy Optimizations <br /> P( (A #8745; B) | I ) <br /> under the constrains <br /> P(I | A) <br /> P(I | B)" )
         Frequencies --> Entropy
         Interpolation --> Entropy
         Entropy --> Bayesian("Bayesian rule <br /> P( I | (A #8745; B)) = P( (A #8745; B) | I ) * P(I)/P(A #8745; B)")
        Bayesian ---> Cleaning("Cleaning <br />inconsistent to <br /> consistent proba ")
        Cleaning -->|Sampling|Sampling[("Population <br /> with Attributes <br /> A,B, I")]