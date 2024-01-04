"""
Starling setup script.

See license in LICENSE.txt.
"""

import setuptools
import os
from bhepop2 import __version__

# short description of the project
DESC = "Synthetic population enrichment from aggregated data"

# get long description from README.md
LONG_DESC = """
A common problem in **generating a representative synthetic population** is that not all attributes of interest are present in the sample.  The purpose is to enrich the synthetic population with additional attributes, after the synthetic population is generated from the original sample.
In many cases, practitioners only have access to aggregated data for important socio-demographic attributes, such as income, education level. 
This package treats the problem to **enrich an initial synthetic population from an aggregated data** provided in the form of a distribution like deciles or quartiles.

Read the [docs](https://bhepop2.readthedocs.io/en/latest/) or see the code and examples on [GitHub](https://github.com/tellae/bhepop2).
"""

# list of classifiers from the PyPI classifiers trove
CLASSIFIERS = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Science/Research",
    "Intended Audience :: Developers",
    "Topic :: Scientific/Engineering",
    "Programming Language :: Python :: 3 :: Only",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "License :: CeCILL-B Free Software License Agreement (CECILL-B)",
]

# only specify install_requires if not in RTD environment
if os.getenv("READTHEDOCS") == "True":
    INSTALL_REQUIRES = []
else:
    with open("requirements.txt") as f:
        INSTALL_REQUIRES = [line.strip() for line in f.readlines()]

# call setup
setuptools.setup(
    name="bhepop2",
    version=__version__,
    license="CECILL-B",
    author="UGE & Tellae",
    author_email="contact@tellae.fr",
    description=DESC,
    long_description_content_type="text/markdown",
    long_description=LONG_DESC,
    url="https://github.com/tellae/bhepop2",
    packages=setuptools.find_packages(),
    classifiers=CLASSIFIERS,
    python_requires=">=3.9",
    install_requires=INSTALL_REQUIRES,
)
