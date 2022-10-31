import numpy as np
import random


def fix_seeds(rnd_seed: int):
    np.random.seed(rnd_seed)
    random.seed(rnd_seed)
