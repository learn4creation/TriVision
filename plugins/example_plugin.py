"""
TriVision — Example Plugin
Demonstrates how to add custom algorithms using the plugin SDK.
This file ships as a reference; it's excluded from auto-loading.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from plugins.sdk import trivision_plugin, Param, ReturnType
import cv2
import numpy as np


@trivision_plugin(
    key="plugin_pencil_hatch",
    label="Pencil Hatching Effect",
    category="Enhancement",
    subcategory="Artistic",
    params=[
        Param.Float("angle", "Hatch angle °", default=45.0, lo=0.0, hi=180.0, step=5.0),
        Param.Int("spacing", "Line spacing", default=4, lo=2, hi=20, step=1),
        Param.Float("sigma", "Edge sigma", default=1.5, lo=0.5, hi=5.0, step=0.5),
    ],
    description="Simulates pencil hatching based on image gradient direction.",
    tags=["artistic", "plugin", "example"],
)
def pencil_hatch(img, angle=45.0, spacing=4, sigma=1.5):
    """Convert an image to a pencil hatching style."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    blurred = cv2.GaussianBlur(gray, (0, 0), sigma)

    # Compute gradient magnitude
    gx = cv2.Sobel(blurred.astype(np.float32), cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(blurred.astype(np.float32), cv2.CV_32F, 0, 1)
    mag = np.sqrt(gx**2 + gy**2)
    mag = (mag / (mag.max() + 1e-10) * 255).astype(np.uint8)

    # Create hatch pattern
    h, w = gray.shape
    canvas = np.ones((h, w), np.uint8) * 255
    rad = np.radians(angle)
    dy, dx = int(spacing * np.sin(rad)), int(spacing * np.cos(rad))
    for y in range(-h, 2 * h, max(1, spacing)):
        x1, y1 = 0, y
        x2, y2 = w, y + int(w * np.tan(rad))
        cv2.line(canvas, (x1, y1), (x2, y2), 0, 1)

    # Blend hatch with edge strength
    alpha = (mag.astype(np.float32) / 255.0)
    result = np.clip(canvas.astype(np.float32) * (1 - alpha * 0.7), 0, 255).astype(np.uint8)
    return cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)


@trivision_plugin(
    key="plugin_neon_edges",
    label="Neon Edge Glow",
    category="Enhancement",
    subcategory="Artistic",
    params=[
        Param.Int("blur_ksize", "Glow radius", default=15, lo=3, hi=51, step=2),
        Param.Choice("color", "Edge color", ["cyan", "magenta", "yellow", "green", "orange"]),
        Param.Int("canny_low", "Canny low", default=50, lo=10, hi=200, step=5),
        Param.Int("canny_high", "Canny high", default=150, lo=50, hi=255, step=5),
    ],
    description="Detects edges and renders them with a vibrant neon glow on a dark background.",
    tags=["artistic", "plugin", "example"],
)
def neon_edges(img, blur_ksize=15, color="cyan", canny_low=50, canny_high=150):
    """Neon glowing edge effect."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    edges = cv2.Canny(gray, canny_low, canny_high)

    color_map = {
        "cyan":    (255, 255, 0),
        "magenta": (255, 0, 255),
        "yellow":  (0, 255, 255),
        "green":   (0, 255, 0),
        "orange":  (0, 165, 255),
    }
    bgr = color_map.get(color, (255, 255, 0))

    canvas = np.zeros((gray.shape[0], gray.shape[1], 3), np.uint8)
    canvas[edges > 0] = bgr

    # Glow via blur + blend
    k = blur_ksize if blur_ksize % 2 == 1 else blur_ksize + 1
    glow = cv2.GaussianBlur(canvas, (k, k), 0)
    result = np.clip(canvas.astype(np.float32) * 1.5 + glow.astype(np.float32), 0, 255)
    return result.astype(np.uint8)
