# Welcome to TriVision

**TriVision** is a unified computer vision and image processing workbench that integrates three powerful libraries — **OpenCV**, **CVIPtools2**, and **scikit-image** — into a single professional desktop application built with PyQt6.

TriVision is designed for students, researchers, and engineers who need to explore, experiment with, and compare image processing algorithms without writing any code. Every algorithm is interactive: change a parameter and see the result instantly.

---

## Table of Contents

1. [Introduction](#introduction)
2. [Getting Started](#getting-started)
3. [Interface Layout](#interface-layout)
4. [Algorithm Tree](#algorithm-tree)
5. [Image Viewers](#image-viewers)
6. [Parameters Panel](#parameters-panel)
7. [Pipeline Builder](#pipeline-builder)
8. [Batch Processing](#batch-processing)
9. [Webcam & Live Recording](#webcam--live-recording)
10. [Algorithm Categories](#algorithm-categories)
    - [Filtering & Smoothing](#filtering--smoothing)
    - [Edge Detection](#edge-detection)
    - [Morphology](#morphology)
    - [Segmentation](#segmentation)
    - [Feature Extraction](#feature-extraction)
    - [Color Processing](#color-processing)
    - [Frequency Domain](#frequency-domain)
    - [Compression](#compression)
    - [Restoration & Enhancement](#restoration--enhancement)
    - [Transforms](#transforms)
    - [TriVision Fusion](#trivision-fusion)
11. [Image I/O](#image-io)
12. [Quality Metrics](#quality-metrics)
13. [A/B Compare & Diff](#ab-compare--diff)
14. [Keyboard Shortcuts](#keyboard-shortcuts)
15. [Plugin SDK](#plugin-sdk)
16. [Settings](#settings)
17. [Themes (Light / Dark)](#themes-light--dark)
18. [Frequently Asked Questions](#frequently-asked-questions)
19. [Troubleshooting](#troubleshooting)

---

## Introduction

### What is TriVision?

TriVision brings together:

| Library | Badge | Speciality |
|---|---|---|
| **OpenCV** | `[CV]` | Speed, camera I/O, DNN inference, real-time processing |
| **CVIPtools2** | `[CL]` | Classical CVIP algorithms, histogram operations, morphology |
| **scikit-image** | `[SK]` | Research-grade algorithms, advanced segmentation, metrics |
| **TriVision** | `[TV]` | Cross-library fusion composites, best-of-breed combinations |

Every algorithm registered in TriVision is available in:
- The **interactive workbench** (select → adjust → preview instantly)
- The **visual pipeline builder** (chain algorithms into workflows)
- The **batch processor** (apply to entire folders of images)
- The **webcam live preview** (apply in real time to camera feed)

### Key Features

- **170+ algorithms** from three industry-standard libraries
- **Live parameter tuning** — every change re-processes immediately (Auto mode)
- **Visual pipeline builder** — drag-and-drop algorithm chains with preset templates
- **Batch processing** — process entire image folders with one click, export CSV/JSON reports
- **Webcam integration** — live feed with real-time algorithm overlay and recording
- **Quality metrics** — automatic PSNR, RMSE, SSIM, and sharpness scores
- **A/B compare** — side-by-side input vs output view
- **Plugin SDK** — extend TriVision with your own algorithms
- **Light / Dark themes** — full theme support with live toggle
- **Configurable settings** — default recording directory and more

---

## Getting Started

### First Launch

When you launch TriVision, the application opens with:
- A **default test image** in the INPUT panel showing geometric shapes used to demonstrate algorithm effects
- The algorithm tree **expanded** on the left, ready to browse
- **Auto mode** enabled — selecting any algorithm immediately processes the input

### Quick Start — Process Your First Image

1. **Load an image**: Go to **File → Open Image** (or press `Ctrl+O`), or click **📂 Load Image** in the Parameters tab on the right.
2. **Browse algorithms**: Use the tree on the left. Click any leaf node (coloured entry) to select an algorithm.
3. **See the result**: The OUTPUT panel updates immediately (Auto mode is on by default).
4. **Adjust parameters**: In the **Parameters** tab on the right, drag sliders or change spinboxes. Output updates in real time.
5. **Save the result**: Go to **File → Save Output** (or press `Ctrl+S`), or click **💾 Save Output** in the Parameters tab.

### Quick Start — Record from Webcam

1. Switch to the **Webcam** tab (right panel).
2. Select your camera index (usually `0`).
3. Click **▶ Start** to begin the live feed.
4. Click **⏺ Record** to start recording. A timestamp-named `.mp4` file is created automatically.
5. Click **■ Stop Record** to stop. A popup will confirm the saved file location and name.

---

## Interface Layout

The TriVision interface is divided into three main regions:

```
┌────────────────────────────────────────────────────────────────────┐
│  Menu Bar: File | View | Pipeline | Help                           │
├───────────────┬────────────────────────────┬───────────────────────┤
│               │   INPUT      │   OUTPUT    │  Parameters           │
│  Algorithm    │   Viewer     │   Viewer    │  Pipeline             │
│  Tree         │   ──────     │   ──────    │  Batch                │
│               │   Histogram  │   Histogram │  Webcam               │
│  [Search]     ├─────────────────────────────┤                       │
│  [Lib Filter] │  ▶ Process  Auto  → Input  │                       │
│               │  ⊕ Diff   A/B   ↺ Reset   │                       │
│               ├─────────────────────────────┤                       │
│               │   Feature Display           │                       │
│               │   Metrics: PSNR RMSE SSIM  │                       │
└───────────────┴─────────────────────────────┴───────────────────────┘
│  Status Bar                                                        │
└────────────────────────────────────────────────────────────────────┘
```

### Menu Bar

| Menu | Items |
|---|---|
| **File** | Open Image (Ctrl+O), Save Output (Ctrl+S), Settings, Exit (Ctrl+Q) |
| **View** | Extract All Features, A/B Compare, Theme submenu |
| **Pipeline** | Run Pipeline, Save Pipeline, Load Pipeline |
| **Help** | Documentation, About TriVision |

### Left Panel — Algorithm Sidebar

Contains:
- **TriVision branding** and theme toggle button
- **Search bar** — type to filter algorithms by name or description
- **Library filter buttons** — filter by All / CV / CL / SK / TV
- **Algorithm Tree** — hierarchical tree of all algorithms, grouped by category and subcategory
- **Algorithm count** display at the bottom

### Centre Panel — Image Workspace

Contains:
- **INPUT viewer** — shows the currently loaded input image
- **OUTPUT viewer** — shows the result of the last processed algorithm
- **Histograms** — live colour channel histograms below each viewer
- **Control row** — Process, Auto, Use as Input, Diff, A/B Compare, Reset buttons
- **Feature Display** — shows extracted feature values (for feature extraction algorithms)
- **Quality Metrics** — PSNR, RMSE, SSIM, Sharpness displayed after each operation

### Right Panel — Tab Widget

Four tabs:
- **Parameters** — algorithm name, library badge, description, parameter controls, and Image I/O buttons
- **Pipeline** — visual pipeline builder
- **Batch** — batch processor
- **Webcam** — live camera feed, recording, live algorithm preview

---

## Algorithm Tree

The algorithm tree on the left is the primary way to browse and select algorithms.

### Structure

Algorithms are organised in a three-level hierarchy:
```
Category (e.g., "Filtering")
  └── Subcategory (e.g., "Smoothing")
        └── [CV] Gaussian Blur
        └── [SK] Bilateral Filter
        └── [CL] Average Filter
```

### Library Badges

Each algorithm is labelled with a coloured badge indicating its source library:
- **[CV]** — OpenCV (blue)
- **[CL]** — CVIPtools2 (green)
- **[SK]** — scikit-image (purple)
- **[TV]** — TriVision fusion (orange)

### Filtering the Tree

**By text**: Type in the search bar above the tree. The tree updates as you type, expanding all matching items.

**By library**: Click one of the coloured filter buttons:
- **All** — show everything
- **CV** — OpenCV only
- **CL** — CVIPtools2 only
- **SK** — scikit-image only
- **TV** — TriVision fusion only

### Right-Click Context Menu

Right-click any algorithm leaf to get:
- **▶ Process now** — immediately process the current input image
- **➕ Add to pipeline** — add this algorithm to the Pipeline builder

### Selecting an Algorithm

Left-click any algorithm leaf. The Parameters tab on the right updates to show:
- Algorithm name
- Library badge with source library name
- Short description
- All adjustable parameters with their controls

If **Auto** mode is on, the algorithm runs immediately on the current input image.

---

## Image Viewers

TriVision has two image viewers in the centre panel:

### INPUT Viewer

- Shows the currently loaded image, webcam snapshot, or default test image
- Scales the image to fit the viewer while preserving aspect ratio
- Below it: a live **histogram** of the input image's colour channels

### OUTPUT Viewer

- Shows the result of the most recently run algorithm or pipeline
- Scales to fit with aspect ratio preserved
- Below it: a live histogram of the output image

### Histograms

Each histogram panel shows:
- **Greyscale images**: a single grey channel histogram
- **Colour images**: overlapping RGB channel histograms (blue=B, green=G, red=R)
- Bin count: 64 bins across the 0–255 intensity range

---

## Parameters Panel

The **Parameters** tab on the right panel is the control centre for algorithm tuning.

### Algorithm Information

At the top of the Parameters tab:
- **Algorithm name** — in bold accent colour
- **Library badge** — e.g., "Library: OpenCV"
- **Description** — a short explanation of what the algorithm does

### Parameter Controls

Each algorithm exposes its tunable parameters as interactive controls:

| Parameter Type | Control | Example |
|---|---|---|
| Integer | Spin box with up/down arrows | Kernel size (1–99) |
| Float | Spin box with decimals | Sigma value (0.01–10.0) |
| Boolean | Checkbox | Use OTSU thresholding |
| Choice | Drop-down combo box | Interpolation method |

**Auto mode**: When the **Auto** checkbox is ticked (default), changing any parameter immediately re-runs the algorithm. Uncheck Auto for manual control, then click **▶ Process** to apply.

### Image I/O Buttons

At the bottom of the Parameters tab:

| Button | Action |
|---|---|
| **📂 Load Image** | Open a file dialog to load any supported image format |
| **💾 Save Output** | Save the current OUTPUT image to disk |
| **📷 Webcam** | Capture a single frame from the default webcam (index 0) |
| **📊 Extract All Features** | Run the TriVision all-features extractor on the current input |

### Supported Image Formats

For loading: PNG, JPEG, JPG, BMP, TIFF, TIF, WebP, PGM, PPM, and all formats supported by OpenCV.
For saving: PNG, JPEG, BMP.

---

## Pipeline Builder

The **Pipeline** tab lets you chain multiple algorithms into a sequential processing workflow.

### How the Pipeline Works

Each algorithm in the pipeline receives the **output of the previous step** as its input. The final result is shown in the OUTPUT viewer.

### Adding Algorithms

Two ways to add an algorithm to the pipeline:
1. **Right-click** any algorithm in the tree → **➕ Add to pipeline**
2. Using the pipeline preset buttons (see below)

### Pipeline List

The pipeline list shows each step with its library badge and name. Steps are processed top-to-bottom.

You can **reorder** steps by dragging them within the list.

### Buttons

| Button | Action |
|---|---|
| **✕ Clear** | Remove all steps from the pipeline |
| **💾** | Save the current pipeline to a JSON file |
| **📂** | Load a pipeline from a JSON file |
| **▶ Run Pipeline** | Process the current input through all pipeline steps |

### Pipeline Presets

Four built-in preset pipelines:

| Preset | Steps included |
|---|---|
| **Edge Detection** | Gaussian Blur → Canny Edge Detection |
| **Denoise + Enhance** | Bilateral Filter → CLAHE → Unsharp Mask |
| **Segmentation** | Gaussian Blur → OTSU Threshold → Morphological Closing |
| **Feature Extraction** | All feature extraction in one pass |

### Saving & Loading Pipelines

Pipelines are saved as JSON files. They store the algorithm key and parameter configuration for each step.

---

## Batch Processing

The **Batch** tab lets you apply any registered algorithm to an entire folder of images at once.

### Setup

1. **Input folder** — browse to the folder containing your source images
2. **Output folder** — browse to where processed images should be saved
3. **Algorithm** — select from the dropdown (all registered algorithms available)
4. **Save outputs** — checkbox to enable/disable saving results to disk

### Running a Batch

Click **▶ Run Batch**. The progress bar fills as each image is processed. The results log shows each filename and its status.

### Export Results

After a batch run, export a report:
- **Export CSV** — saves a `.csv` with per-image statistics (processing time, status)
- **Export JSON** — saves a `.json` with full batch metadata

### Supported Input Files

Batch processing recognises: PNG, JPG, JPEG, BMP, TIFF, TIF, WebP, PGM, PPM.

---

## Webcam & Live Recording

The **Webcam** tab provides live camera feed with real-time algorithm preview and video recording.

### Controls

| Control | Description |
|---|---|
| **Camera** spin box | Camera device index (0 = default camera, 1, 2 … for additional cameras) |
| **FPS display** | Live frames-per-second counter and resolution |
| **▶ Start** | Open the camera and begin streaming |
| **■ Stop** | Stop the camera and close the device |
| **📷 Snapshot** | Capture the current frame and send it to the INPUT panel |
| **⏺ Record** | Start recording to MP4 |
| **■ Stop Record** | Stop the current recording |

### Starting the Camera

Click **▶ Start**. TriVision opens the camera using DirectShow (on Windows) for better compatibility. The live feed appears in the viewer at up to 1280×720 at 30 FPS.

### Taking Snapshots

Click **📷 Snapshot** at any time to send the current live frame to the **INPUT panel** in the main workbench. You can then immediately run any algorithm on it.

### Recording Video

1. Start the camera (▶ Start)
2. Click **⏺ Record** — recording begins immediately. A timestamp-named file is created automatically:
   - Default location: `~/Videos/TriVision/` (or your configured recording directory)
   - Filename format: `recording_YYYYMMDD_HHMMSS.mp4`
3. Click **■ Stop Record** (or stop the camera) to finish recording.
4. A **save notification popup** appears confirming the file location and name.

### Changing the Default Recording Directory

Go to **File → Settings** and set a new default recording folder. All future recordings will save there.

### Live Algorithm Preview

The **Live Algorithm Preview** group at the bottom of the Webcam tab lets you apply any image algorithm to the live feed in real time:

1. Select an algorithm from the dropdown
2. Click **▶ Live Preview** to enable
3. The live feed is processed through the selected algorithm
4. Click **■ Stop Preview** to return to the raw camera feed

The live preview uses a thread pool so the camera feed never stalls.

### Recording With Live Preview

If Live Preview is active during recording, the **processed** (algorithm-applied) frames are recorded, not the raw camera feed. This lets you capture edge-detected, filtered, or otherwise processed video directly.

---

## Algorithm Categories

TriVision organises its 170+ algorithms into categories. Here is a detailed overview of each.

---

### Filtering & Smoothing

Filtering is the process of modifying image intensities by applying a mathematical operation at each pixel, typically using a kernel (sliding window) convolution.

#### Average / Box Filter `[CV]` `[CL]`
Replaces each pixel with the average of its neighbourhood. Fast and simple, but blurs edges.
- **Kernel Size** — the width/height of the averaging window (must be odd)

#### Gaussian Blur `[CV]` `[SK]`
Applies a Gaussian-weighted average. Pixels closer to the centre have more influence. Produces smoother, more natural blurring than box filter. Standard pre-processing step before edge detection.
- **Kernel Size** — window size (must be odd)
- **Sigma X / Y** — standard deviation in horizontal and vertical directions

#### Median Filter `[CV]` `[SK]`
Replaces each pixel with the median of its neighbourhood. Excellent for removing **salt-and-pepper noise** while preserving edges better than Gaussian blur.
- **Kernel Size** — neighbourhood size

#### Bilateral Filter `[CV]` `[SK]`
Applies smoothing while preserving strong edges by considering both spatial distance and intensity similarity. Ideal for noise reduction while keeping details sharp.
- **Diameter** — pixel neighbourhood diameter
- **Sigma Colour** — filter sigma in colour space
- **Sigma Space** — filter sigma in coordinate space

#### Non-Local Means Denoising `[CV]`
Powerful denoising algorithm that averages similar patches from across the entire image (not just local neighbourhood). Superior quality for photographic images.
- **h** — filter strength (larger = more denoising, less detail)
- **Template Window Size** — patch size for comparison
- **Search Window Size** — region to search for similar patches

#### Guided Filter `[CV]` `[SK]`
Edge-preserving smoothing filter that uses a "guide image" (usually the input itself) to steer the smoothing. Very fast and produces natural-looking results.

---

### Edge Detection

Edge detection identifies boundaries in an image where intensity changes significantly.

#### Canny Edge Detector `[CV]` `[SK]`
The classic multi-stage edge detector. First applies Gaussian smoothing, then computes image gradients, performs non-maximum suppression, and applies hysteresis thresholding for clean, thin edges.
- **Low Threshold** — lower bound for hysteresis
- **High Threshold** — upper bound (edges above this are always kept)
- **Aperture Size** — Sobel kernel size for gradient computation

#### Sobel Operator `[CV]` `[SK]`
Computes first-order image derivatives using Sobel kernels. Can detect edges in X direction, Y direction, or combined magnitude. Less noise-resistant than Canny.
- **ksize** — kernel size
- **Direction** — X, Y, or Combined magnitude

#### Laplacian of Gaussian (LoG) `[CV]` `[SK]`
Applies Gaussian smoothing then computes second-order derivatives (Laplacian). Zero-crossings in the result mark edges.
- **ksize** — Laplacian kernel size

#### Prewitt Operator `[SK]`
Similar to Sobel but uses equal weights. Detects edges in horizontal and vertical directions.

#### Roberts Cross `[SK]`
Uses 2×2 cross-shaped kernels, very fast. Sensitive to noise. Good for fine detail detection.

#### Scharr Filter `[CV]` `[SK]`
An optimised version of the Sobel operator with better rotational symmetry. Preferred when gradient accuracy is critical.

#### Holistically-nested Edge Detection (HED) `[CV]`
Deep learning-based edge detection using a convolutional neural network. Produces semantically meaningful edges rather than just intensity gradients.

---

### Morphology

Morphological operations process images based on shape, using a **structuring element** (kernel).

#### Erosion `[CV]` `[CL]`
Shrinks bright regions and expands dark regions. Removes small white noise and thin protrusions.
- **Kernel Size** — structuring element size
- **Iterations** — number of erosion passes

#### Dilation `[CV]` `[CL]`
Expands bright regions and shrinks dark regions. Fills small holes in foreground objects.
- **Kernel Size** — structuring element size
- **Iterations** — number of dilation passes

#### Opening `[CV]` `[CL]`
Erosion followed by dilation. Removes small objects (noise) from the foreground while preserving the shape of larger objects.

#### Closing `[CV]` `[CL]`
Dilation followed by erosion. Fills small holes and gaps in foreground objects.

#### Morphological Gradient `[CV]`
The difference between dilation and erosion. Produces an outline of the objects.

#### Top Hat `[CV]`
Difference between the input and its opening. Highlights bright regions that are smaller than the structuring element.

#### Black Hat `[CV]`
Difference between the closing and the input. Highlights dark regions smaller than the structuring element.

#### Skeletonization `[SK]`
Reduces binary objects to a one-pixel-wide centrelined skeleton while preserving topology (connectivity).

---

### Segmentation

Segmentation divides an image into meaningful regions.

#### OTSU Thresholding `[CV]` `[SK]`
Automatically finds the optimal global threshold by maximising between-class variance. Ideal when the image has a bimodal intensity histogram.

#### Adaptive Thresholding `[CV]`
Applies different threshold values across local regions of the image. Better than global thresholding for images with varying lighting.
- **Method** — Mean or Gaussian weighted local neighbourhood
- **Block Size** — local neighbourhood size
- **C** — constant subtracted from the mean

#### K-Means Colour Segmentation `[CV]`
Clusters image pixels into K colour groups. Each pixel is assigned to the nearest cluster centroid.
- **K** — number of colour clusters (2–16)

#### Watershed Segmentation `[CV]` `[SK]`
Treats the image gradient as a topographic surface and "floods" from pre-marked seed points. Separates touching objects reliably.

#### SLIC Superpixels `[SK]`
Segments the image into approximately uniform superpixels using a modified K-means in colour+coordinate space. Useful as a pre-processing step for many tasks.
- **n_segments** — approximate number of superpixels
- **Compactness** — trade-off between colour and spatial proximity

#### Felzenszwalb Segmentation `[SK]`
Efficient graph-based image segmentation. Produces regions with internally consistent colour/texture.
- **Scale** — higher = larger regions
- **Sigma** — pre-smoothing sigma
- **Min Size** — minimum component size

#### GrabCut `[CV]`
Interactive foreground/background segmentation. Automatically separates the image into foreground and background using an iterative graph-cut algorithm.

#### Mean Shift `[CV]`
Non-parametric segmentation based on density estimation. Finds modes in the colour-space distribution. Does not require specifying K.

---

### Feature Extraction

Feature extraction computes numerical descriptors that characterise image content.

#### ORB Keypoints `[CV]`
Oriented FAST and Rotated BRIEF — fast, patent-free keypoint detection and description. Detects corners and computes binary descriptors.
- **n_keypoints** — maximum number of keypoints to detect

#### SIFT Keypoints `[CV]`
Scale-Invariant Feature Transform — detects keypoints at multiple scales and computes gradient-based descriptors. Invariant to scale, rotation, and illumination.

#### HOG Descriptor `[SK]`
Histogram of Oriented Gradients. Computes gradient orientation histograms in local cells. The foundation of classic pedestrian detection (DPM, SVM).
- **Orientations** — number of gradient orientation bins
- **Pixels per Cell** — cell size

#### LBP (Local Binary Pattern) `[SK]`
Local Binary Pattern texture descriptor. Compares each pixel to its circular neighbourhood and generates a binary code. Fast and effective for texture classification.

#### Hu Moments `[CV]`
Seven mathematically derived moment invariants that are invariant to translation, rotation, and scale. Used for shape matching.

#### Harris Corners `[CV]`
Detects image corners by analysing the second-order structure of the gradient. Classic corner detector for feature matching and structure-from-motion.
- **Block Size** — neighbourhood size
- **k** — Harris detector free parameter

#### Statistical Features `[TV]`
TriVision fusion that computes a complete set of statistical texture features: mean, variance, standard deviation, skewness, kurtosis, entropy — per channel.

#### All Features `[TV]`
Runs the complete TriVision feature extraction pipeline in one step: colour moments, texture statistics, edge density, shape descriptors, HOG summary, and LBP histogram. Results appear in the Feature Display panel.

---

### Color Processing

Algorithms that operate on the colour representation of images.

#### Colour Space Conversion `[CV]`
Convert between colour spaces: BGR, RGB, HSV, HSL, LAB, YCrCb, XYZ, Greyscale.
- **Target Space** — the output colour space

#### Channel Split & Merge `[CV]`
Splits a colour image into individual channels (B, G, R or H, S, V) for visualisation or individual channel processing.

#### Colour Histogram Equalisation `[CV]` `[CL]`
Equalises the histogram of each colour channel independently to stretch contrast. Can introduce colour shifts; use CLAHE for better results.

#### CLAHE — Contrast Limited Adaptive Histogram Equalisation `[CV]`
Applies histogram equalisation in small tiles across the image with contrast limiting to prevent over-amplification of noise. Superior to global equalisation.
- **Clip Limit** — contrast limiting threshold
- **Tile Grid Size** — number of tiles in X and Y

#### Gamma Correction `[CV]` `[TV]`
Applies a power-law transformation: `output = input^gamma`. Gamma < 1 brightens, gamma > 1 darkens.
- **Gamma** — power-law exponent

#### White Balance `[TV]`
Adjusts the colour balance of an image to remove colour casts from the light source. Uses the grey-world assumption.

#### Colour Transfer `[TV]`
Transfers the colour statistics (mean and standard deviation in LAB colour space) from a reference image to the target. Useful for colour grading.

#### Hue Rotation `[CV]` `[TV]`
Rotates the hue channel of an image in HSV space, effectively shifting all colours around the colour wheel.
- **Hue Shift** — degrees to rotate hue (0–180 in OpenCV's 8-bit HSV)

---

### Frequency Domain

Processing images in the frequency domain using Fourier transforms allows global operations like filtering without spatial convolution.

#### DFT / FFT `[CV]` `[SK]`
Computes the Discrete Fourier Transform of an image. The magnitude spectrum is displayed (log scale). Bright regions in the centre represent low-frequency content; edges represent high frequencies.

#### FFT Low-Pass Filter `[CV]` `[SK]`
Removes high-frequency content (fine detail, noise) from the image in the frequency domain. Equivalent to spatial domain blurring but with a sharp frequency cutoff.
- **Cutoff** — fraction of the spectrum to keep (0.0–1.0)

#### FFT High-Pass Filter `[CV]` `[SK]`
Removes low-frequency content (background, large uniform regions), leaving only fine detail and edges.
- **Cutoff** — fraction of the spectrum to remove

#### Band-Pass Filter `[CV]` `[TV]`
Keeps a band of frequencies between a low and high cutoff. Preserves mid-frequency texture while removing both noise and background.

#### Notch Filter `[TV]`
Removes specific frequency components from the spectrum. Used to remove periodic noise patterns (e.g., grid patterns from scanning).

#### DCT (Discrete Cosine Transform) `[CV]` `[SK]`
Computes the Discrete Cosine Transform, the basis of JPEG compression. Energy is concentrated in the upper-left corner of the DCT coefficient matrix.

---

### Compression

TriVision includes implementations of several image compression algorithms for educational comparison.

#### JPEG Quantisation Simulation `[CV]` `[TV]`
Simulates JPEG compression at a specified quality level. Applies DCT-based block quantisation and shows the visual artefacts that result.
- **Quality** — 1 (lowest) to 95 (highest)

#### Run-Length Encoding (RLE) `[TV]`
Lossless compression of greyscale images using run-length encoding. Effective for images with large uniform regions.

#### Block Truncation Coding (BTC) `[TV]`
Divides the image into blocks and quantises each block to two intensity levels. A classic early lossy compression scheme.
- **Block Size** — typically 4×4 or 8×8
- **Bits per Pixel** — output bit depth per block

#### Huffman Coding (Visualisation) `[TV]`
Builds and applies a Huffman code table to image pixel values. Shows the compression ratio achieved by entropy coding alone.

#### Wavelet Compression `[SK]`
Applies Discrete Wavelet Transform (DWT) based compression. More efficient than DCT at high compression ratios with fewer blocking artefacts.
- **Level** — number of decomposition levels
- **Threshold** — wavelet coefficient threshold for zeroing

---

### Restoration & Enhancement

These algorithms improve image quality by correcting degradations or enhancing specific characteristics.

#### Unsharp Masking `[CV]` `[SK]`
Sharpens an image by subtracting a blurred version from the original and adding it back scaled. Controls: amount, radius, threshold.
- **Amount** — how much sharpening to apply
- **Radius** — blur radius used for the mask
- **Threshold** — minimum edge difference to sharpen

#### Deconvolution / Wiener Filter `[SK]`
Attempts to reverse the blurring caused by a known point spread function (PSF). The Wiener filter balances deblurring against noise amplification.

#### Inpainting `[CV]`
Fills in damaged/missing regions of an image using the pixels around the boundary of the mask. Uses either the Navier-Stokes algorithm or the fast marching method.
- **Radius** — inpainting neighbourhood radius
- **Method** — NS (Navier-Stokes) or Telea (fast marching)

#### Retinex (MSRCR) `[TV]`
Multi-Scale Retinex with Colour Restoration. Attempts to separate reflectance from illumination, producing images that appear to be taken with ideal, uniform lighting. Similar to what the human visual system does naturally.

#### Histogram Stretching `[CV]` `[CL]`
Linearly stretches the intensity range to span the full 0–255 range. Simple contrast enhancement for underexposed images.

#### Noise Addition `[CV]` `[SK]`
Adds Gaussian, Salt-and-Pepper, or Speckle noise to an image. Useful for testing noise robustness of algorithms.
- **Noise Type** — Gaussian, S&P, Speckle

#### Super-Resolution (EDSR) `[CV]`
Uses a pre-trained deep neural network to upscale images while reconstructing detail. Significantly better than interpolation methods.
- **Scale** — upscaling factor (2, 3, or 4)

---

### Transforms

Geometric and intensity transformations.

#### Affine Transform `[CV]` `[SK]`
Applies a 2D affine transformation (rotation, translation, scaling, shearing) to an image.
- **Angle** — rotation in degrees
- **Scale** — uniform scaling factor
- **Tx, Ty** — translation in X and Y pixels

#### Perspective Transform `[CV]`
Applies a projective (homography) transformation. Changes the apparent viewpoint.

#### Resize `[CV]` `[SK]`
Resizes the image to specified dimensions using Nearest, Bilinear, Bicubic, Lanczos, or Area interpolation.
- **Width, Height** — target size in pixels
- **Method** — interpolation algorithm

#### Flip `[CV]`
Flips horizontally, vertically, or both.

#### Rotation `[CV]` `[SK]`
Rotates the image around its centre by a specified angle.

#### Crop `[TV]`
Crops a rectangular region from the image using relative coordinates (0.0–1.0).

#### Padding `[CV]` `[TV]`
Adds border pixels around the image. Border types: constant colour, replicate edge, reflect.

---

### TriVision Fusion

TriVision fusion algorithms (`[TV]`) combine techniques from multiple libraries to produce results that no single library can achieve alone.

#### Smart Denoise `[TV]`
Automatically selects and combines the best denoising approach based on image analysis: non-local means for photographic content, bilateral for structured images.

#### Auto Enhance `[TV]`
Analyses the input image and applies an appropriate combination of CLAHE, gamma correction, and sharpening to improve visual quality automatically.

#### Style Transfer (Artistic) `[TV]`
Applies artistic style transforms using a combination of colour transfer and texture synthesis. Produces painterly or sketch-like effects.

#### Document Scan Simulation `[TV]`
Simulates the look of a scanned document: applies perspective correction estimate, decolourises, applies adaptive thresholding and sharpening, and adjusts contrast.

#### Depth from Defocus `[TV]`
Estimates a rough depth/focus map by analysing the local sharpness variation across the image using multi-scale Laplacian responses.

#### Panorama Stitch (Preview) `[TV]`
Provides a visual demonstration of feature-based image stitching using ORB features and RANSAC homography estimation.

---

## Image I/O

### Loading Images

**From menu**: File → Open Image (Ctrl+O)
**From button**: Parameters tab → 📂 Load Image
**From webcam**: Webcam tab → 📷 Snapshot (sends frame to INPUT)
**Drag-and-drop**: Drag an image file directly onto the INPUT viewer

Supported formats: PNG, JPEG, JPG, BMP, TIFF, TIF, WebP, PGM, PPM

### Saving Images

**From menu**: File → Save Output (Ctrl+S)
**From button**: Parameters tab → 💾 Save Output

Saves the current OUTPUT image. Supported formats: PNG, JPEG, BMP.

Greyscale images are saved correctly without colour space issues.

### Using Output as New Input

Click **→ Use as Input** to send the current OUTPUT image to the INPUT viewer. This allows chaining operations manually without using the pipeline builder. When combined with Auto mode, the next selected algorithm immediately processes the new input.

---

## Quality Metrics

After every processing operation, TriVision automatically computes four quality metrics comparing the INPUT and OUTPUT images. These are displayed in the metrics bar below the control row.

| Metric | Full Name | What It Measures |
|---|---|---|
| **PSNR** | Peak Signal-to-Noise Ratio | Overall signal quality, in decibels. Higher = better. Typical range: 20–50 dB |
| **RMSE** | Root Mean Square Error | Pixel-level difference magnitude. Lower = better. 0 = identical |
| **SSIM** | Structural Similarity Index | Perceptual similarity (0–1). 1 = identical. Better than PSNR for perceived quality |
| **Sharpness** | Laplacian Variance | High-frequency energy in the output. Higher = sharper |

**Note**: Metrics are computed in greyscale (luminance only). If images have different sizes, the output is resized to match the input for comparison.

---

## A/B Compare & Diff

### A/B Compare

Click **A/B Compare** (or View → A/B Compare) to display the INPUT and OUTPUT side by side in the OUTPUT viewer, separated by a blue divider line. The left half shows the INPUT labelled "INPUT" and the right shows the OUTPUT labelled "OUTPUT".

This is useful for:
- Quickly comparing "before" and "after" processing
- Demonstrating the effect of an algorithm to someone else

### Diff View

Click **⊕ Diff** to display a ×3 amplified absolute difference between the INPUT and OUTPUT. Every pixel in the result shows `3 × |input - output|`. Blue regions indicate near-identical areas; bright white/coloured regions show where the algorithm has made large changes.

This is useful for:
- Understanding exactly which pixels were modified by an algorithm
- Detecting subtle changes in restoration algorithms
- Quality checking after compression/decompression

---

## Keyboard Shortcuts

| Shortcut | Action |
|---|---|
| `Ctrl+O` | Open Image |
| `Ctrl+S` | Save Output |
| `Ctrl+Q` | Quit TriVision |
| `Ctrl+T` | Toggle Light/Dark theme |
| `Enter` / `Return` | Process (when Auto is off) |

---

## Plugin SDK

TriVision supports extending the algorithm library with custom plugins. Plugins are Python files placed in the `plugins/` directory.

### Plugin File Structure

Create a new `.py` file in `D:\TriVision\plugins\`. TriVision scans this folder on startup.

### Minimal Plugin Example

```python
# plugins/my_invert.py

from core.registry import register, AlgorithmSpec, Param, Lib, ReturnType
import numpy as np

def my_invert_fn(img, strength=1.0):
    \"\"\"Inverts the image with adjustable strength.\"\"\"
    inverted = 255 - img
    return cv2.addWeighted(img, 1.0 - strength, inverted, strength, 0)

def register_plugin():
    register(AlgorithmSpec(
        key         = "my_invert",
        label       = "My Invert",
        lib         = Lib.TRIVISION,
        category    = "Custom",
        subcategory = "Effects",
        fn          = my_invert_fn,
        params      = [
            Param.Float("strength", "Strength", default=1.0, lo=0.0, hi=1.0, step=0.05),
        ],
        return_type  = ReturnType.IMAGE,
        description  = "Inverts the image colours with adjustable strength.",
    ))
```

### Plugin Requirements

- The file must define a `register_plugin()` function
- Must import and use `register()` from `core.registry`
- Must use `AlgorithmSpec` with a **unique key** (duplicate keys overwrite existing algorithms)
- The `fn` callable must accept `(img: np.ndarray, **kwargs)` and return an image or features dict

### Parameter Types

| Factory | Python Type | Widget |
|---|---|---|
| `Param.Int(name, label, default, lo, hi, step)` | int | QSpinBox |
| `Param.Float(name, label, default, lo, hi, step)` | float | QDoubleSpinBox |
| `Param.Bool(name, label, default)` | bool | QCheckBox |
| `Param.Choice(name, label, choices, default)` | str | QComboBox |

### Plugin Loading

Plugins are loaded at startup. If a plugin fails to load, the error is printed to the console and TriVision continues. Failed plugins do not crash the application.

---

## Settings

### Opening Settings

Go to **File → Settings** in the menu bar.

### Available Settings

#### Default Recording Directory

The folder where video recordings from the Webcam tab are saved. Default: `~/Videos/TriVision/`

- Click **Browse…** to open a folder picker dialog
- The current path is shown in the dialog
- Click **OK** to save the new path. All future recordings use the new directory.
- Click **Cancel** to keep the existing path unchanged.

### Settings Persistence

Settings are stored in `~/.trivision/trivision_settings.json`. This file is created automatically on first launch.

---

## Themes (Light / Dark)

TriVision supports two themes with a complete, seamless switch.

### Switching Themes

- **Theme toggle button** — in the top-left of the algorithm sidebar (shows ☀ Light or 🌙 Dark)
- **View → Theme → Light Theme** or **Dark Theme** in the menu
- **View → Theme → System OS Match** — detects your OS theme and matches it
- **Ctrl+T** keyboard shortcut

### Theme Scope

Themes apply to every part of the interface:
- Menu bar and menus
- Algorithm tree (background, hover, selection)
- Image viewers and histograms
- All buttons, spinboxes, comboboxes, checkboxes
- Parameters panel, pipeline list, batch results area
- Webcam controls
- Status bar
- Feature display

---

## Frequently Asked Questions

**Q: Why does the output look the same as the input?**
A: Some algorithms (e.g., small kernel Gaussian blur) produce subtle changes. Try the Diff view (⊕ Diff button) to see what changed, amplified ×3.

**Q: Can I use TriVision without a webcam?**
A: Yes. The Webcam tab requires a camera for live feed, but all other features (image processing, batch, pipeline) work without any camera.

**Q: Does TriVision support GPU acceleration?**
A: TriVision uses OpenCV's standard CPU implementations. If your OpenCV build includes CUDA support, OpenCV may use GPU internally for some operations. Most operations use optimised multi-core CPU code.

**Q: How do I add my own algorithm?**
A: See the [Plugin SDK](#plugin-sdk) section. Create a Python file in the `plugins/` folder with a `register_plugin()` function.

**Q: Can I process RAW camera files?**
A: TriVision uses OpenCV's `imread()` which does not support RAW formats. Convert your RAW files to TIFF or PNG first using your camera software or tools like dcraw or libraw.

**Q: The pipeline runs but shows a blank output — why?**
A: Ensure your pipeline's first step receives an appropriate input type. Some algorithms expect greyscale input; if your source is RGB, add a greyscale conversion step at the beginning.

**Q: Why does my recording look fast-forwarded?**
A: TriVision compensates for gaps in frame delivery by writing duplicate frames to maintain the target 30 FPS. If the camera's actual FPS is lower than 30, duplicate frames are inserted to avoid speed-up. This is expected behaviour.

**Q: How do I export features to a file?**
A: After extracting features (📊 Extract All Features or running a feature extraction algorithm), the Batch processor can export feature values as CSV or JSON.

**Q: Is TriVision open source?**
A: TriVision is distributed under the license in the `LICENSE` file in the root of the repository.

---

## Troubleshooting

### Camera Does Not Open

- Ensure no other application is using the camera
- Try camera index 0, 1, 2 in the Camera spin box
- On Windows, TriVision uses DirectShow. If the camera is not detected, check Device Manager
- Restart the application and try again

### Processing Is Very Slow

- Switch to the **Light theme** (sometimes GPU compositing in dark mode adds overhead)
- Disable **Auto** mode and process manually with `▶ Process`
- Use a smaller input image for testing

### Application Crashes on CV Super-Resolution

- The EDSR model requires the OpenCV `contrib` module. If you see a crash here, your OpenCV build may not include DNN Superres support. The algorithm will error gracefully in most cases.

### Plugin Not Loading

- Check the console output for the error message from the plugin loader
- Ensure the plugin file has a `register_plugin()` function (exact name)
- Verify there are no syntax errors in your plugin file

### Recording File is Empty or Corrupt

- Ensure you have write permission to the recording directory
- If the disk is full, recording will fail silently. Check available disk space.
- Always use **■ Stop Record** before closing the application to flush the video file header. If TriVision crashes during recording, the last recording may be corrupt.

### Windows: Qt Platform Plugin Error

- Ensure Qt6 DLLs are in the system PATH or the application directory
- Run `Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser` in PowerShell, then reinstall PyQt6

### Settings File Corrupt

- Delete `~/.trivision/trivision_settings.json` and restart. The file will be recreated with defaults.

---

*TriVision — Unified Computer Vision & Image Processing Workbench*
*OpenCV · CVIPtools2 · scikit-image · TriVision Fusion*
