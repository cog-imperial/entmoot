[![Tests](https://github.com/cog-imperial/entmoot/actions/workflows/python-package.yml/badge.svg?branch=entmoot-v2)](https://github.com/cog-imperial/entmoot/actions/workflows/python-package.yml)
[![codecov](https://codecov.io/gh/cog-imperial/entmoot/branch/entmoot-v2/graph/badge.svg)](https://codecov.io/gh/cog-imperial/entmoot)
![Python versions](https://img.shields.io/badge/python-3.7%20%7C%203.8%20%7C%203.9%20%7C%203.10%20%7C%203.11-blue)
[![Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black) 
[![Read the Docs](https://readthedocs.org/projects/entmoot/badge/?version=latest)](https://entmoot.readthedocs.io/en/latest/)

<img src="media/entmoot_logo.png" width="400">

`ENTMOOT` (**EN**semble **T**ree **MO**del **O**ptimization **T**ool) is a framework to perform Bayesian Optimization using tree-based surrogate models. Gradient-boosted tree models from `lightgbm` are combined with a distance-based uncertainty 
measure in a deterministic global optimization framework to optimize black-box functions. More
details on the method here: https://arxiv.org/abs/2003.04774.

## Documentation

The docs can be found here: https://entmoot.readthedocs.io/

## How to reference ENTMOOT

When using any `ENTMOOT` for any publications please reference this software package as:
```
@article{thebelt2021entmoot,
  title={ENTMOOT: A framework for optimization over ensemble tree models},
  author={Thebelt, Alexander and Kronqvist, Jan and Mistry, Miten and Lee, Robert M and Sudermann-Merx, Nathan and Misener, Ruth},
  journal={Computers \& Chemical Engineering},
  volume={151},
  pages={107343},
  year={2021},
  publisher={Elsevier}
}
```


## Authors
* **[Alexander Thebelt](https://optimisation.doc.ic.ac.uk/person/alexander-thebelt/)** ([ThebTron](https://github.com/ThebTron)) - Imperial College London
* **[Nathan Sudermann-Merx](https://www.mannheim.dhbw.de/profile/sudermann-merx)** ([spiralulam](https://github.com/spiralulam)) - Cooperative State University Mannheim
## License
The ENTMOOT package is released under the BSD 3-Clause License. Please refer to the [LICENSE](https://github.com/cog-imperial/entmoot/blob/master/LICENSE) file for details.

## Acknowledgements
The support of BASF SE, Lugwigshafen am Rhein is gratefully acknowledged.
