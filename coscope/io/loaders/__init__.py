"""Dataset loaders — implement AbstractLoader."""

from coscope.io.loaders.base_loader import BaseLoader
from coscope.io.loaders.gsm8k_loader import GSM8KLoader
from coscope.io.loaders.hotpot_loader import HotpotLoader
from coscope.io.loaders.math_loader import MathLoader
from coscope.io.loaders.musique_loader import MusiqueLoader
from coscope.io.loaders.wiki_loader import WikiLoader


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
