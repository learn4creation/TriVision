# TriVision

**The unified Computer Vision & Image Processing Workbench** — OpenCV, CVIPtools2, and scikit-image in one application.

```
pip install -r requirements.txt
python main.py
```
## Development
Built with assistance from Claude (Anthropic).
Wraps OpenCV (BSD), scikit-image (BSD), and CVIPtools algorithms.
```
```
TriVision/
├── LICENSE          
├── README.md        
├── requirements.txt
├── main.py
├── core/
├── pipeline/
├── batch/
└── plugins/
---

## Why TriVision?

| Library | Strength | Weakness |
|---|---|---|
| **OpenCV** | Speed, camera I/O, DNN, production | Limited research algorithms |
| **CVIPtools2** | Classical CVIP curriculum, freq-domain, compression | Windows-only original |
| **scikit-image** | Research-grade algorithms, correct metrics, SSIM | Slower, Python-only |
| **TriVision** | **All three, unified, with pipeline chaining** | — |

---

## Algorithm Count: 130+

### Analysis (`core/algorithms_opencv.py` + `core/algorithms_skimage.py`)

| Category | Algorithms |
|---|---|
| **Edge Detection** | Sobel, Prewitt, Roberts, Laplacian, Canny, Kirsch, Hough (lines + circles), LoG (Marr-Hildreth), Frangi vessel, Sato tubular, Meijering neurite |
| **Segmentation** | Threshold, Otsu, Adaptive, K-Means, Watershed, GrabCut, SLIC superpixels, Felzenszwalb, Quickshift, Chan-Vese, Active Contour, Fuzzy C-Means |
| **Morphology** | Erode, Dilate, Open, Close, Gradient, Top Hat, Black Hat, Skeleton, Connected Labels, Binary Closing (skimage), Remove Small Objects, Convex Hull |
| **Transforms** | DFT (magnitude + phase), DCT, Haar, Walsh-Hadamard, Wavelet subbands |
| **Feature Detection** | ORB, AKAZE, BRISK, Harris, Shi-Tomasi, FAST, SIFT, Template Matching |
| **Texture** | LBP, HOG (visualised), DAISY (visualised), GLCM features |
| **Feature Extraction** | Histogram stats, RST-invariant Hu moments, Laws texture energy, Region properties, Spectral ring/sector features |
| **Object Detection** | Face (Haar), Eye (Haar), QR code |

### Enhancement (`core/algorithms_cvip_fusion.py` + `core/algorithms_opencv.py`)

| Category | Algorithms |
|---|---|
| **Histogram** | Equalize, CLAHE, Slide, Stretch, Hyperbolize, Adaptive Hist Eq (skimage) |
| **Tonal** | Gamma correction, Sigmoid contrast, Intensity rescale |
| **Spatial** | Unsharp mask, Bilateral, NL Means, Adaptive Contrast (ACE), Laplacian sharpen, High-boost |
| **Frequency Domain** | Butterworth LP/HP/BP/BR, HFE, Notch, Homomorphic |
| **Computational Photo** | Detail enhance, Stylization, Pencil sketch, NLM denoising |
| **Pseudo-Color** | JET/HOT/VIRIDIS/TURBO/MAGMA colormaps, Intensity slicing, Frequency color map |

### Restoration (`core/algorithms_cvip_fusion.py` + `core/algorithms_skimage.py`)

| Category | Algorithms |
|---|---|
| **Noise Models** | Gaussian, Salt & Pepper (6 types) |
| **Denoising** | Wavelet denoise (skimage), Total Variation (Chambolle), Hybrid pipeline (bilateral+TV+NLM) |
| **Deconvolution** | Wiener classic, Constrained Least Squares, Richardson-Lucy, Unsupervised Wiener |
| **Inpainting** | Biharmonic inpainting (skimage) |
| **Geometric** | Rotate, Scale, Translate, Shear, Barrel/Pincushion, Sinusoidal Warp, Perspective |

### Compression (`core/algorithms_cvip_fusion.py`)

| Algorithm | Unique feature |
|---|---|
| Zonal DCT | Keep low-freq zone; adjustable fraction |
| Threshold DCT | Keep largest N% coefficients |
| Block Truncation Coding | Preserves 1st + 2nd statistical moments |
| Vector Quantization | LBG/k-means codebook on image blocks |
| Wavelet Threshold | Multi-level Haar + coefficient thresholding |
| DPCM | Differential predictive coding with N-bit quantisation |
| JPEG | Via OpenCV; quality 1–100 |
| WebP | Via OpenCV; quality 1–100 |

### TriVision Fusion (`core/algorithms_cvip_fusion.py`)

Algorithms that combine all three libraries:

| Algorithm | What it does |
|---|---|
| **Multi-Scale Edge Fusion** | Canny + Sobel + Frangi + LoG fused into one richer edge map |
| **Hybrid Denoise Pipeline** | Bilateral → TV-Chambolle → NLM, removes noise at multiple scales |
| **Comprehensive Feature Extraction** | Histogram + Hu + Laws + GLCM + region props + spectral in one call |
| **Image Quality Score** | Sharpness + contrast + SNR + SSIM — all in one dict |
| **Multi-Method Segmentation** | SLIC (skimage) + K-means (OpenCV) consensus boundaries overlay |

---

## GUI Features

- **Algorithm tree** — 130+ algorithms organised by Analysis / Enhancement / Restoration / Compression, with library badges [CV] [CL] [SK] [TV]
- **Search bar** — real-time fuzzy search across all algorithm names and descriptions
- **Library filter** — show only OpenCV, CVIPtools2, scikit-image, or TriVision algorithms
- **Auto-process** — live preview as parameters change
- **Live histograms** — RGB/gray channel histograms under both panels
- **A/B Compare** — side-by-side input vs output with a single click
- **Diff View** — ×3 amplified pixel difference
- **Quality Metrics** — PSNR, RMSE, SSIM (if skimage available), Sharpness after every operation
- **Feature Panel** — displays all extracted feature values in monospace

### Visual Pipeline Builder (Pipeline tab)
- Add algorithms by right-clicking the tree → "Add to pipeline"
- Reorder via drag-and-drop
- One-click preset pipelines: Edge Detection, Denoise+Enhance, Segmentation, Feature Extraction
- Save/load pipelines as JSON
- "Run Pipeline" applies the full chain end-to-end

### Batch Processor (Batch tab)
- Point at an input folder → process all images → save outputs
- CSV and JSON export with per-image PSNR, RMSE, SSIM, Sharpness, timing
- Cancellable mid-run

### Plugin SDK (plugins/)
Drop a `.py` file in `plugins/` using `@trivision_plugin`:

```python
from plugins.sdk import trivision_plugin, Param

@trivision_plugin(
    key="my_algo",
    label="My Algorithm",
    category="Analysis",
    subcategory="Edge Detection",
    params=[Param.Float("sigma", "Sigma", 2.0, 0.1, 10.0, 0.1)],
)
def my_algo(img, sigma=2.0):
    ...
    return result_img
```
Restart TriVision — your algorithm appears in the tree, gets auto-generated parameter controls, and works in the pipeline builder and batch processor.

---

## Project Structure

```
trivision/
├── main.py                          # PyQt6 application
├── requirements.txt
├── README.md
├── core/
│   ├── __init__.py
│   ├── registry.py                  # AlgorithmSpec, Param, AlgorithmRegistry
│   ├── algorithms_opencv.py         # ~50 OpenCV algorithms
│   ├── algorithms_skimage.py        # ~40 scikit-image algorithms
│   └── algorithms_cvip_fusion.py    # ~35 CVIPtools2 + TriVision fusion
├── pipeline/
│   ├── __init__.py
│   └── engine.py                    # Pipeline, PipelineNode, preset pipelines
├── batch/
│   ├── __init__.py
│   └── processor.py                 # BatchProcessor, BatchReport
└── plugins/
    ├── __init__.py
    ├── sdk.py                        # @trivision_plugin decorator + load_plugins()
    └── example_plugin.py             # Two example plugin algorithms
```

---

## Improvements Over Any Single Tool

| Feature | OpenCV alone | CVIPtools | scikit-image | **TriVision** |
|---|---|---|---|---|
| Algorithms | ~50 relevant | ~70 | ~40 | **130+** |
| Cross-library pipelines | No | No | No | **Yes** |
| Visual pipeline builder | No | No | No | **Yes** |
| Batch + CSV report | Manual | No | No | **Yes** |
| Plugin SDK | No | No | No | **Yes** |
| SSIM metric | No | No | Yes | **Yes** |
| Registration required | No | Yes | No | **No** |
| Cross-platform | Yes | No | Yes | **Yes** |
| A/B comparison | No | No | No | **Yes** |
| Live histograms | No | No | No | **Yes** |
