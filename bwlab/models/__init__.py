"""Image-only model interface and native SAM3 adapter."""
from ._base import Model
from .sam3 import Sam3GridAdapter

__all__ = ["Model", "Sam3GridAdapter"]
