"""Installed extension registration, run before loading a model or starting work."""

from importlib.metadata import entry_points

_LOADED = set()


def load_extensions():
    """Entry points in trainfer.extensions are zero-argument registrars.

    Only installed, trusted Python packages participate. HTTP requests cannot
    choose import paths. Failure stops startup rather than serving with a
    partially available optimization method. Restart after fixing a plugin.
    """
    for entry in sorted(entry_points(group="trainfer.extensions"), key=lambda ep: ep.name):
        identity = (entry.name, entry.value)
        if identity in _LOADED:
            continue
        registrar = entry.load()
        if not callable(registrar):
            raise TypeError(f"extension {entry.name!r} must resolve to a callable registrar")
        registrar()
        _LOADED.add(identity)
