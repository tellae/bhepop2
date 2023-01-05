"""
Starling setup script.

See license in LICENSE.txt.
"""

import setuptools
import os
from src import __version__

# short description of the project
DESC = "TODO"

# get long description from README.md
# with open("README.md", "r") as fh:
#     LONG_DESC = fh.read()
LONG_DESC = r"""
TODO
"""

# list of classifiers from the PyPI classifiers trove
CLASSIFIERS = [
    "TODO",
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
    license="TODO",
    author="TODO",
    author_email="TODO",
    description=DESC,
    long_description_content_type="text/x-rst",
    long_description=LONG_DESC,
    url="https://github.com/tellae/synthetic-pop-uge-tellae",
    packages=setuptools.find_packages(),
    classifiers=CLASSIFIERS,
    python_requires=">=3.6",
    install_requires=INSTALL_REQUIRES,
)
