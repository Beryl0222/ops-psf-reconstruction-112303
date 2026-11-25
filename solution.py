"""Conventional solution-module alias re-exporting :mod:`psf_reconstruction`."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from psf_reconstruction import *  # noqa: F403,E402
import psf_reconstruction as _impl  # noqa: E402

__all__ = getattr(_impl, "__all__", [])
for _name in __all__:
    globals()[_name] = getattr(_impl, _name)


if __name__ == "__main__":
    raise SystemExit(_impl.main())
