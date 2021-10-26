from bo import algorithm, metric, plot

__version__ = "0.3.7"
__all__ = ["algorithm", "metric", "plot"]

try:
    import torch_tools  # noqa: F401

    __all__.append("torch_tools")
except ImportError:
    pass
