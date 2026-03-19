"""
TriVision — Batch Processor

Applies any pipeline or single algorithm to a folder of images.
Generates a CSV report with per-image quality metrics.
Supports progress callbacks for GUI integration.
"""

from __future__ import annotations
import csv
import json
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional
import cv2
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pipeline.engine import Pipeline
from core.registry import REGISTRY


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif",
                         ".webp", ".pgm", ".ppm", ".exr"}


@dataclass
class BatchResult:
    filename: str
    success: bool
    psnr: Optional[float] = None
    rmse: Optional[float] = None
    ssim: Optional[float] = None
    sharpness: Optional[float] = None
    compression_ratio: Optional[float] = None
    bpp: Optional[float] = None
    elapsed_ms: Optional[float] = None
    error: str = ""
    output_path: str = ""


@dataclass
class BatchReport:
    pipeline_name: str
    total: int = 0
    succeeded: int = 0
    failed: int = 0
    results: list[BatchResult] = field(default_factory=list)

    def summary(self) -> dict:
        ok = [r for r in self.results if r.success]
        if not ok:
            return {"total": self.total, "succeeded": 0, "failed": self.failed}
        psnrs = [r.psnr for r in ok if r.psnr is not None]
        ssims = [r.ssim for r in ok if r.ssim is not None]
        speeds = [r.elapsed_ms for r in ok if r.elapsed_ms is not None]
        return {
            "total": self.total,
            "succeeded": self.succeeded,
            "failed": self.failed,
            "avg_psnr_dB": round(sum(psnrs)/len(psnrs), 2) if psnrs else None,
            "avg_ssim": round(sum(ssims)/len(ssims), 4) if ssims else None,
            "avg_ms_per_image": round(sum(speeds)/len(speeds), 1) if speeds else None,
        }

    def to_csv(self, path: str):
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=[
                "filename","success","psnr","rmse","ssim","sharpness",
                "compression_ratio","bpp","elapsed_ms","error","output_path"])
            w.writeheader()
            for r in self.results:
                w.writerow({
                    "filename": r.filename, "success": r.success,
                    "psnr": r.psnr, "rmse": r.rmse, "ssim": r.ssim,
                    "sharpness": r.sharpness, "compression_ratio": r.compression_ratio,
                    "bpp": r.bpp, "elapsed_ms": r.elapsed_ms,
                    "error": r.error, "output_path": r.output_path,
                })

    def to_json(self, path: str):
        with open(path, "w") as f:
            json.dump({
                "pipeline": self.pipeline_name,
                "summary": self.summary(),
                "results": [r.__dict__ for r in self.results],
            }, f, indent=2)


def _compute_metrics(original: np.ndarray, processed: np.ndarray) -> dict:
    """Compute PSNR, RMSE, SSIM (if skimage available), sharpness."""
    def to_gray(img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape)==3 else img

    g_orig = to_gray(original).astype(np.float64)
    g_proc = to_gray(processed).astype(np.float64)

    if g_orig.shape != g_proc.shape:
        g_proc = cv2.resize(g_proc.astype(np.uint8),
                             (g_orig.shape[1], g_orig.shape[0])).astype(np.float64)

    mse = float(np.mean((g_orig - g_proc)**2))
    rmse = float(np.sqrt(mse))
    psnr = float(10*np.log10(255**2/(mse+1e-10)))
    sharpness = float(cv2.Laplacian(processed, cv2.CV_64F).var())

    ssim = None
    try:
        from skimage.metrics import structural_similarity
        ssim = float(structural_similarity(g_orig, g_proc, data_range=255.0))
    except ImportError:
        pass

    return {"psnr": round(psnr,2), "rmse": round(rmse,2),
            "ssim": round(ssim,4) if ssim is not None else None,
            "sharpness": round(sharpness,2)}


class BatchProcessor:
    """
    Processes a folder of images through a pipeline or single algorithm.
    """

    def __init__(self,
                  pipeline: Optional[Pipeline] = None,
                  algo_key: Optional[str] = None,
                  algo_params: dict = None):
        if pipeline is None and algo_key is None:
            raise ValueError("Provide either a pipeline or an algo_key.")
        self.pipeline = pipeline
        self.algo_key = algo_key
        self.algo_params = algo_params or {}
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def run(self,
             input_dir: str,
             output_dir: str,
             save_outputs: bool = True,
             progress_cb: Optional[Callable[[int, int, str], None]] = None
             ) -> BatchReport:
        """
        Process all images in input_dir.

        progress_cb(current, total, filename) is called for each image.
        """
        self._cancel = False
        in_path = Path(input_dir)
        out_path = Path(output_dir)
        if save_outputs:
            out_path.mkdir(parents=True, exist_ok=True)

        files = sorted([f for f in in_path.iterdir()
                        if f.suffix.lower() in SUPPORTED_EXTENSIONS])

        name = self.pipeline.name if self.pipeline else (
            REGISTRY.get(self.algo_key).label if REGISTRY.get(self.algo_key) else self.algo_key)
        report = BatchReport(pipeline_name=name, total=len(files))

        for i, fpath in enumerate(files):
            if self._cancel:
                break
            if progress_cb:
                progress_cb(i, len(files), fpath.name)

            t0 = time.perf_counter()
            img = cv2.imread(str(fpath))
            if img is None:
                report.results.append(BatchResult(fpath.name, False, error="Cannot read image"))
                report.failed += 1
                continue

            try:
                if self.pipeline:
                    result = self.pipeline.final_output(img)
                else:
                    spec = REGISTRY.get(self.algo_key)
                    if spec is None:
                        raise ValueError(f"Unknown algo: {self.algo_key}")
                    result = spec.fn(img, **self.algo_params)
                    if isinstance(result, tuple):
                        result = result[0]

                elapsed = (time.perf_counter() - t0) * 1000

                # Coerce to BGR image
                if isinstance(result, np.ndarray):
                    out_img = result
                elif isinstance(result, dict):
                    out_img = img  # feature dict — keep original
                else:
                    out_img = img

                metrics = _compute_metrics(img, out_img)
                out_fname = ""
                if save_outputs and isinstance(out_img, np.ndarray):
                    out_fname = str(out_path / fpath.name)
                    cv2.imwrite(out_fname, out_img)

                br = BatchResult(
                    filename=fpath.name, success=True,
                    psnr=metrics["psnr"], rmse=metrics["rmse"],
                    ssim=metrics["ssim"], sharpness=metrics["sharpness"],
                    elapsed_ms=round(elapsed,1), output_path=out_fname,
                )
                report.results.append(br)
                report.succeeded += 1

            except Exception as e:
                elapsed = (time.perf_counter() - t0) * 1000
                report.results.append(BatchResult(
                    fpath.name, False,
                    elapsed_ms=round(elapsed,1),
                    error=str(e),
                ))
                report.failed += 1

        if progress_cb:
            progress_cb(len(files), len(files), "Done")
        return report
