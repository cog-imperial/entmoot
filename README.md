[![Tests](https://github.com/cog-imperial/entmoot/actions/workflows/python-package.yml/badge.svg?branch=entmoot-v2)](https://github.com/cog-imperial/entmoot/actions/workflows/python-package.yml)

<img src="media/entmoot_logo.png" width="400">

`ENTMOOT` (**EN**semble **T**ree **MO**del **O**ptimization **T**ool) is a framework to handle tree-based surrogate models in Bayesian optimization applications. Gradient-boosted
tree models from `lightgbm` are combined with a distance-based uncertainty 
measure in a deterministic global optimization framework to optimize black-box functions. More
details on the method here: https://arxiv.org/abs/2003.04774.

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
* **[David Walz](https://www.linkedin.com/in/walzds/?originalSubdomain=de)** ([DavidWalz](https://github.com/DavidWalz)) - BASF SE
## License
The ENTMOOT package is released under the BSD 3-Clause License. Please refer to the [LICENSE](https://github.com/cog-imperial/entmoot/blob/master/LICENSE) file for details.

## Acknowledgements
The support of BASF SE, Lugwigshafen am Rhein is gratefully acknowledged.
