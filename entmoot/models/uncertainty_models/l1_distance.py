from entmoot.models.uncertainty_models.base_distance import NonCatDistance


class L1Distance(NonCatDistance):

    def __init__(self, dist_trafo, acq_sense):
        raise NotImplementedError()
