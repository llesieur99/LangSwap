"""Reuse model holders within a stage; release the pool between GPU stages."""
from threading import RLock

_POOL: dict = {}
_LOCK = RLock()


def get_or_create(key, factory):
    """Return a cached instance for ``key``, creating it via ``factory`` once."""
    with _LOCK:
        inst = _POOL.get(key)
        if inst is None:
            inst = factory()
            _POOL[key] = inst
        return inst


def clear_pool():
    """Drop cached engines so VRAM can be reused by the next stage (24 GB card)."""
    with _LOCK:
        _POOL.clear()
