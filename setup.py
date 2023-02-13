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
with open("README.md", "r") as fh:
    LONG_DESC = fh.read()

# list of classifiers from the PyPI classifiers trove
CLASSIFIERS = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Science/Research",
    "Intended Audience :: Developers",
    "Topic :: Scientific/Engineering",
    "Programming Language :: Python :: 3.8",
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
    python_requires=">=3.8",
    install_requires=INSTALL_REQUIRES,
)
