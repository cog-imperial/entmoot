import os.path

from setuptools import setup


def get_version(package_name):
    here = os.path.abspath(os.path.dirname(__file__))
    fp = os.path.join(here, f"{package_name}/__init__.py")
    for line in open(fp).readlines():
        if line.startswith("__version__"):
            return line.split('"')[1]


setup(
    name="basf-bo",
    version=get_version("bo"),
    description="Multiobjective Bayesian Optimization",
    url="https://gitlab.roqs.basf.net/bayesopt/brain",
    packages=["bo"],
    install_requires=[
        "basf-opti",
        "click",
        "numpy",
        "pandas",
        "plotly",
        "pymoo",
        "pyrff",
        "scikit-learn",
        "tqdm",
    ],
    tests_require=["pytest"],
    extras_require={
        "botorch": ["botorch==0.3.3", "gpytorch==1.3.0"],
        "entmoot": ["entmoot>=0.2.2", "gurobipy"],
        "pyreto": ["basf-pyreto"],
    },
    python_requires=">=3.6",
)
