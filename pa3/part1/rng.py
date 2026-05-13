"""Random number generator with context management.

Lets us deterministically initialize replicated parameters (e.g. the router)
identically across ranks while still drawing rank-specific noise where we need
it (e.g. for the per-rank expert in expert parallelism).
"""
from contextlib import contextmanager
import numpy as np

_registered_rngs = {}
current_rng = np.random.RandomState(0)


def register_rng(name, rng=None):
    global _registered_rngs
    if rng is None:
        rng = np.random.RandomState(0)
    _registered_rngs[name] = rng


def get_rng():
    global current_rng
    return current_rng


@contextmanager
def rng_context(name):
    global _registered_rngs
    global current_rng
    prev_context = current_rng
    if name not in _registered_rngs:
        register_rng(name)
    current_rng = _registered_rngs[name]
    try:
        yield
    finally:
        current_rng = prev_context


register_rng("expert")
register_rng("router")
register_rng("testing")
