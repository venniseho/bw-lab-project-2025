"""Array functions behind the Compose-compatible stimulus transforms."""
from .fence import generate_fence as draw_fence
from .fragmentation import generate_fragmentation as fragment_contours

__all__ = ["draw_fence", "fragment_contours"]
