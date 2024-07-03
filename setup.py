from pathlib import Path

from setuptools import find_packages, setup

project_root = Path(__file__).resolve().parent

about = {}
version_path = project_root / "entmoot" / "__version__.py"
with version_path.open() as f:
    exec(f.read(), about)

setup(
    name="entmoot",
    author=about["__author__"],
    author_email=about["__author_email__"],
    license=about["__license__"],
    version=about["__version__"],
    url="https://github.com/cog-imperial/entmoot",
    packages=find_packages(exclude=["tests", "docs"]),
    install_requires=[
        "numpy<=2.0.0",
        "lightgbm>=4.0.0",
        "gurobipy",
        "pyomo==6.7.0"
    ],
    setup_requires=[
        "numpy<=2.0.0",
        "lightgbm>=4.0.0",
        "gurobipy",
        "pyomo==6.7.0"
    ],
)
