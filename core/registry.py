"""
TriVision — Algorithm Registry
Unified abstraction layer over OpenCV, CVIPtools2, and scikit-image.

Every algorithm is described by an AlgorithmSpec:
  - which library it comes from
  - what parameters it accepts (with ranges, defaults, types)
  - what categories it belongs to
  - what it returns (image, features dict, metrics dict, or tuple)

The registry is the single source of truth for the GUI, pipeline engine,
batch processor, and plugin SDK.
"""

from __future__ import annotations
import enum
from dataclasses import dataclass, field
from typing import Any, Callable, Optional


# ─── Library tags ────────────────────────────────────────────────────────────

class Lib(enum.Enum):
    OPENCV  = "OpenCV"
    CVIP    = "CVIPtools2"
    SKIMAGE = "scikit-image"
    TRIVISION = "TriVision"   # algorithms that fuse multiple libraries


# ─── Return type tags ────────────────────────────────────────────────────────

class ReturnType(enum.Enum):
    IMAGE    = "image"       # np.ndarray
    FEATURES = "features"    # dict[str, float]
    METRICS  = "metrics"     # dict[str, float]
    OVERLAY  = "overlay"     # np.ndarray (BGR overlay on input)
    TUPLE    = "tuple"       # (image, ratio, bpp) — compression algorithms


# ─── Parameter descriptor ────────────────────────────────────────────────────

@dataclass
class Param:
    """Describes one tunable parameter of an algorithm."""
    name: str
    label: str
    kind: str           # "int" | "float" | "bool" | "choice"
    default: Any
    lo: Any = None
    hi: Any = None
    step: Any = None
    choices: list[str] = field(default_factory=list)
    tip: str = ""

    @staticmethod
    def Int(name, label, default, lo=0, hi=255, step=1, tip="") -> "Param":
        return Param(name, label, "int", default, lo, hi, step, tip=tip)

    @staticmethod
    def Float(name, label, default, lo=0.0, hi=1.0, step=0.01, tip="") -> "Param":
        return Param(name, label, "float", default, lo, hi, step, tip=tip)

    @staticmethod
    def Bool(name, label, default=True, tip="") -> "Param":
        return Param(name, label, "bool", default, tip=tip)

    @staticmethod
    def Choice(name, label, choices, default=None, tip="") -> "Param":
        return Param(name, label, "choice", default or choices[0], choices=choices, tip=tip)


# ─── Algorithm specification ─────────────────────────────────────────────────

@dataclass
class AlgorithmSpec:
    """Full descriptor for one algorithm."""
    key: str                        # unique snake_case identifier
    label: str                      # display name
    lib: Lib                        # source library
    category: str                   # top-level category
    subcategory: str                # sub-category
    fn: Callable                    # the actual function
    params: list[Param]             # tunable parameters
    return_type: ReturnType = ReturnType.IMAGE
    description: str = ""
    tags: list[str] = field(default_factory=list)


# ─── Registry ────────────────────────────────────────────────────────────────

class AlgorithmRegistry:
    """Central registry holding all algorithm specs."""

    def __init__(self):
        self._specs: dict[str, AlgorithmSpec] = {}

    def register(self, spec: AlgorithmSpec):
        self._specs[spec.key] = spec

    def get(self, key: str) -> Optional[AlgorithmSpec]:
        return self._specs.get(key)

    def all(self) -> list[AlgorithmSpec]:
        return list(self._specs.values())

    def by_category(self) -> dict[str, dict[str, list[AlgorithmSpec]]]:
        """Returns {category: {subcategory: [specs]}}."""
        out: dict[str, dict[str, list[AlgorithmSpec]]] = {}
        for spec in self._specs.values():
            out.setdefault(spec.category, {}).setdefault(spec.subcategory, []).append(spec)
        return out

    def by_lib(self, lib: Lib) -> list[AlgorithmSpec]:
        return [s for s in self._specs.values() if s.lib == lib]

    def search(self, query: str) -> list[AlgorithmSpec]:
        q = query.lower()
        return [s for s in self._specs.values()
                if q in s.label.lower() or q in s.description.lower()
                or any(q in t for t in s.tags)]

    def __len__(self): return len(self._specs)


# ─── Global instance ─────────────────────────────────────────────────────────

REGISTRY = AlgorithmRegistry()


def register(spec: AlgorithmSpec):
    """Convenience function used by all algorithm modules."""
    REGISTRY.register(spec)
