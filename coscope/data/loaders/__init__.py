"""Loader registry for coscope.data."""

from coscope.data.loaders.base_loader import BaseLoader
from coscope.data.loaders.musique_loader import MusiqueLoader
from coscope.data.loaders.wiki_loader import WikiLoader
from coscope.data.loaders.hotpot_loader import HotpotLoader
from coscope.data.loaders.gsm8k_loader import GSM8KLoader
from coscope.data.loaders.math_loader import MathLoader


LOADER_REGISTRY = {
    "musique": MusiqueLoader,
    "2wikimhqa": WikiLoader,
    "2wikimultihopqa": WikiLoader,   # alias for legacy config key
    "hotpotqa": HotpotLoader,
    "gsm8k": GSM8KLoader,
    "math": MathLoader,
}


def get_loader(dataset: str, *, data_dir, split: str, dataset_config=None) -> BaseLoader:
    """Instantiate the loader class for `dataset`."""
    cls = LOADER_REGISTRY.get(dataset)
    if cls is None:
        raise KeyError(f"Unknown dataset: {dataset!r}. Known: {sorted(LOADER_REGISTRY)}")
    return cls(data_dir=data_dir, split=split, dataset_config=dataset_config)


__all__ = [
    "BaseLoader",
    "MusiqueLoader",
    "WikiLoader",
    "HotpotLoader",
    "GSM8KLoader",
    "MathLoader",
    "LOADER_REGISTRY",
    "get_loader",
]
