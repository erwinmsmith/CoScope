"""Dataset loaders — implement AbstractLoader."""

from dataio.loaders.base_loader import BaseLoader
from dataio.loaders.gsm8k_loader import GSM8KLoader
from dataio.loaders.hotpot_loader import HotpotLoader
from dataio.loaders.math_loader import MathLoader
from dataio.loaders.musique_loader import MusiqueLoader
from dataio.loaders.wiki_loader import WikiLoader


_REGISTRY = {
    "gsm8k":      GSM8KLoader,
    "hotpotqa":   HotpotLoader,
    "math":       MathLoader,
    "musique":    MusiqueLoader,
    "2wikimhqa":  WikiLoader,
}


def get_loader(dataset: str, **kwargs):
    key = dataset.lower()
    if key not in _REGISTRY:
        raise ValueError(f"Unknown dataset '{dataset}'. Available: {list(_REGISTRY)}")
    return _REGISTRY[key](**kwargs)


__all__ = [
    "BaseLoader", "GSM8KLoader", "HotpotLoader", "MathLoader",
    "MusiqueLoader", "WikiLoader", "get_loader",
]
