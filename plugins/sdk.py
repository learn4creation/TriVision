"""
TriVision — Plugin SDK

Researchers and developers can add new algorithms by:
1. Creating a .py file in the plugins/ directory
2. Defining a function decorated with @trivision_plugin
3. Restarting TriVision — the algorithm auto-appears in the GUI tree

Example plugin file (plugins/my_custom_edge.py):

    from plugins.sdk import trivision_plugin, Param

    @trivision_plugin(
        key="my_edge",
        label="My Custom Edge Detector",
        category="Analysis",
        subcategory="Edge Detection",
        params=[
            Param.Float("sigma", "Sigma", default=2.0, lo=0.1, hi=10.0, step=0.1),
            Param.Int("threshold", "Threshold", default=50, lo=0, hi=255, step=5),
        ],
        description="My edge detector combining Gaussian smoothing and thresholding.",
        tags=["custom", "edges"],
    )
    def my_edge(img, sigma=2.0, threshold=50):
        import cv2, numpy as np
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape)==3 else img
        blurred = cv2.GaussianBlur(gray, (0,0), sigma)
        edges = cv2.Canny(blurred, threshold//2, threshold)
        return edges
"""

from __future__ import annotations
import importlib
import importlib.util
import sys
from pathlib import Path
from typing import Callable, Optional
from functools import wraps

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.registry import REGISTRY, AlgorithmSpec, Param, Lib, ReturnType, register


# Re-export for plugin authors
__all__ = ["trivision_plugin", "Param", "ReturnType", "load_plugins"]


def trivision_plugin(
    key: str,
    label: str,
    category: str,
    subcategory: str,
    params: list[Param] = None,
    return_type: ReturnType = ReturnType.IMAGE,
    description: str = "",
    tags: list[str] = None,
):
    """
    Decorator that registers a function as a TriVision algorithm.

    Usage:
        @trivision_plugin(key="my_algo", label="My Algorithm",
                          category="Analysis", subcategory="Edge Detection")
        def my_algo(img, sigma=1.0):
            ...
            return result_img
    """
    def decorator(fn: Callable) -> Callable:
        spec = AlgorithmSpec(
            key=key,
            label=label,
            lib=Lib.TRIVISION,
            category=category,
            subcategory=subcategory,
            fn=fn,
            params=params or [],
            return_type=return_type,
            description=description,
            tags=tags or ["plugin"],
        )
        REGISTRY.register(spec)

        @wraps(fn)
        def wrapper(*args, **kwargs):
            return fn(*args, **kwargs)
        return wrapper
    return decorator


def load_plugins(plugin_dir: str = None) -> list[str]:
    """
    Scan plugin_dir for .py files (excluding __init__ and sdk) and import them.
    Defaults to a plugins/ folder next to this file (works from any cwd).
    Returns list of successfully loaded plugin filenames.
    """
    if plugin_dir is None:
        plugin_dir = Path(__file__).parent
    plugin_path = Path(plugin_dir)
    if not plugin_path.exists():
        return []

    loaded = []
    for py_file in sorted(plugin_path.glob("*.py")):
        if py_file.stem in ("__init__", "sdk", "example_plugin"):
            continue
        try:
            spec = importlib.util.spec_from_file_location(
                f"trivision_plugin_{py_file.stem}", py_file)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            loaded.append(py_file.name)
            print(f"  [Plugin] Loaded: {py_file.name}")
        except Exception as e:
            print(f"  [Plugin] ERROR loading {py_file.name}: {e}")

    return loaded
