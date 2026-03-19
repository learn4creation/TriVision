"""
TriVision — scikit-image Algorithm Registrations
Algorithms uniquely strong in skimage: SLIC, active contours, Frangi,
region props, SSIM, LoG, template matching, inpainting, texture, etc.
"""

import cv2
import numpy as np
from .registry import REGISTRY, AlgorithmSpec, Param, Lib, ReturnType

_L = Lib.SKIMAGE
_I = ReturnType.IMAGE
_F = ReturnType.FEATURES
_O = ReturnType.OVERLAY
_M = ReturnType.METRICS


def _gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img

def _norm8(a):
    a = a.astype(np.float64)
    mn, mx = a.min(), a.max()
    return ((a - mn) / (mx - mn + 1e-10) * 255).astype(np.uint8)

def _bgr(img):
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if len(img.shape) == 2 else img

def _float(img):
    return img.astype(np.float64) / 255.0


def _try_skimage(fn):
    """Decorator that catches ImportError gracefully."""
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except ImportError as e:
            h, w = args[0].shape[:2]
            out = args[0].copy()
            cv2.putText(out, f"scikit-image not installed: {e}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            return out
    return wrapper


# ═══ SEGMENTATION ══════════════════════════════════════════════════════════════

@_try_skimage
def _slic(img, n_segments=100, compactness=10.0):
    from skimage.segmentation import slic, mark_boundaries
    out = slic(_float(_bgr(img)), n_segments=n_segments, compactness=compactness,
               channel_axis=2, start_label=1)
    marked = mark_boundaries(_float(_bgr(img)), out)
    return (np.clip(marked, 0, 1) * 255).astype(np.uint8)

@_try_skimage
def _felzenszwalb(img, scale=100.0, sigma=0.5, min_size=50):
    from skimage.segmentation import felzenszwalb, mark_boundaries
    segs = felzenszwalb(_float(_bgr(img)), scale=scale, sigma=sigma, min_size=min_size, channel_axis=2)
    marked = mark_boundaries(_float(_bgr(img)), segs)
    return (np.clip(marked, 0, 1) * 255).astype(np.uint8)

@_try_skimage
def _quickshift(img, kernel_size=3, max_dist=6, ratio=0.5):
    from skimage.segmentation import quickshift, mark_boundaries
    segs = quickshift(_float(_bgr(img)), kernel_size=kernel_size, max_dist=max_dist,
                      ratio=ratio, channel_axis=2)
    marked = mark_boundaries(_float(_bgr(img)), segs)
    return (np.clip(marked, 0, 1) * 255).astype(np.uint8)

@_try_skimage
def _chan_vese(img, mu=0.25, lambda1=1.0, lambda2=1.0, max_iter=200):
    from skimage.segmentation import chan_vese
    gray = _float(_gray(img))
    cv_result = chan_vese(gray, mu=mu, lambda1=lambda1, lambda2=lambda2,
                          max_num_iter=max_iter, extended_output=False)
    return (cv_result.astype(np.uint8) * 255)

@_try_skimage
def _active_contour(img, alpha=0.015, beta=10.0):
    from skimage.segmentation import active_contour
    gray = _float(_gray(img))
    h, w = gray.shape
    # Circular initial contour
    t = np.linspace(0, 2*np.pi, 400)
    snake = np.array([h/2 + h/3*np.sin(t), w/2 + w/3*np.cos(t)]).T
    try:
        snake_out = active_contour(cv2.GaussianBlur(gray, (0,0), 3), snake,
                                    alpha=alpha, beta=beta)
        result = _bgr(img).copy()
        pts = np.intp(snake_out[:,::-1])
        cv2.polylines(result, [pts], True, (0,255,0), 2)
        return result
    except Exception:
        return _bgr(img)


# ═══ EDGE DETECTION (skimage-specific) ════════════════════════════════════════

@_try_skimage
def _log_edge(img, sigma=2.0):
    from skimage.filters import laplace, gaussian
    gray = _float(_gray(img))
    smoothed = gaussian(gray, sigma=sigma)
    edges = laplace(smoothed)
    return _norm8(np.abs(edges))

@_try_skimage
def _frangi(img, sigmas_start=1, sigmas_stop=10, sigmas_num=2):
    from skimage.filters import frangi
    gray = _float(_gray(img))
    sigmas = np.linspace(sigmas_start, sigmas_stop, sigmas_num)
    result = frangi(gray, sigmas=sigmas)
    return _norm8(result)

@_try_skimage
def _sato(img):
    from skimage.filters import sato
    gray = _float(_gray(img))
    return _norm8(sato(gray))

@_try_skimage
def _meijering(img):
    from skimage.filters import meijering
    gray = _float(_gray(img))
    return _norm8(meijering(gray))


# ═══ RESTORATION ═══════════════════════════════════════════════════════════════

@_try_skimage
def _wavelet_denoise(img, sigma=None, mode='soft', wavelet='db1'):
    from skimage.restoration import denoise_wavelet
    f = _float(_bgr(img))
    result = denoise_wavelet(f, sigma=sigma, mode=mode, wavelet=wavelet,
                              channel_axis=2, convert2ycbcr=True, rescale_sigma=True)
    return (np.clip(result, 0, 1) * 255).astype(np.uint8)

@_try_skimage
def _tv_denoise(img, weight=0.1):
    from skimage.restoration import denoise_tv_chambolle
    f = _float(_bgr(img))
    result = denoise_tv_chambolle(f, weight=weight, channel_axis=2)
    return (np.clip(result, 0, 1) * 255).astype(np.uint8)

@_try_skimage
def _bm3d_denoise(img, sigma=25.0):
    try:
        from skimage.restoration import denoise_wavelet
        # BM3D not always available; fall back to wavelet
        return _wavelet_denoise(img, sigma=sigma/255.0)
    except Exception:
        return img

@_try_skimage
def _inpaint_biharmonic(img, radius=5):
    from skimage.restoration import inpaint_biharmonic
    gray = _float(_gray(img))
    h, w = gray.shape
    # Create mask from very dark pixels
    mask = (gray < 0.05).astype(bool)
    if mask.sum() == 0:
        mask[h//2-radius:h//2+radius, w//2-radius:w//2+radius] = True
    result = inpaint_biharmonic(gray, mask)
    return _norm8(result)

@_try_skimage
def _richardson_lucy(img, psf_size=5, iterations=10):
    from skimage.restoration import richardson_lucy
    from skimage.filters import gaussian
    gray = _float(_gray(img))
    psf = np.ones((psf_size, psf_size), dtype=np.float64) / (psf_size**2)
    degraded = cv2.filter2D(gray, -1, psf)
    result = richardson_lucy(degraded, psf, num_iter=iterations)
    return _norm8(result)

@_try_skimage
def _unsupervised_wiener(img):
    from skimage.restoration import unsupervised_wiener
    from scipy.signal import convolve2d
    gray = _float(_gray(img))
    psf = np.ones((5,5)) / 25.0
    deconvolved, _ = unsupervised_wiener(gray, psf)
    return _norm8(deconvolved)


# ═══ ENHANCEMENT ═══════════════════════════════════════════════════════════════

@_try_skimage
def _equalize_adapthist(img, clip_limit=0.01, kernel_size=None):
    from skimage.exposure import equalize_adapthist
    f = _float(_gray(img))
    result = equalize_adapthist(f, clip_limit=clip_limit)
    return _norm8(result)

@_try_skimage
def _adjust_gamma(img, gamma=1.0):
    from skimage.exposure import adjust_gamma
    f = _float(_bgr(img))
    result = adjust_gamma(f, gamma=gamma)
    return (np.clip(result,0,1)*255).astype(np.uint8)

@_try_skimage
def _adjust_sigmoid(img, cutoff=0.5, gain=10.0):
    from skimage.exposure import adjust_sigmoid
    f = _float(_bgr(img))
    result = adjust_sigmoid(f, cutoff=cutoff, gain=gain)
    return (np.clip(result,0,1)*255).astype(np.uint8)

@_try_skimage
def _rescale_intensity(img, plo=2, phi=98):
    from skimage.exposure import rescale_intensity
    f = _float(_gray(img))
    p_lo, p_hi = np.percentile(f, (plo, phi))
    result = rescale_intensity(f, in_range=(p_lo, p_hi))
    return _norm8(result)

@_try_skimage
def _match_histograms(img):
    from skimage.exposure import match_histograms
    f = _float(_bgr(img))
    # Match to a uniform histogram
    reference = np.random.RandomState(0).uniform(0,1,f.shape)
    result = match_histograms(f, reference, channel_axis=2)
    return (np.clip(result,0,1)*255).astype(np.uint8)


# ═══ MORPHOLOGY (skimage extras) ═══════════════════════════════════════════════

@_try_skimage
def _binary_closing(img, radius=3):
    from skimage.morphology import binary_closing, disk
    gray = _gray(img)
    binary = gray > 127
    result = binary_closing(binary, disk(radius))
    return (result.astype(np.uint8) * 255)

@_try_skimage
def _remove_small_objects(img, min_size=64):
    from skimage.morphology import remove_small_objects
    gray = _gray(img)
    binary = gray > 127
    result = remove_small_objects(binary, min_size=min_size)
    return (result.astype(np.uint8) * 255)

@_try_skimage
def _convex_hull(img):
    from skimage.morphology import convex_hull_image
    gray = _gray(img)
    binary = gray > 127
    result = convex_hull_image(binary)
    out = _bgr(img).copy()
    out[result & ~binary] = [0, 200, 100]
    return out


# ═══ TEXTURE & FEATURES ════════════════════════════════════════════════════════

@_try_skimage
def _local_binary_pattern(img, P=8, R=1.0, method='uniform'):
    from skimage.feature import local_binary_pattern
    gray = _float(_gray(img))
    lbp = local_binary_pattern(gray, P=P, R=R, method=method)
    return _norm8(lbp)

@_try_skimage
def _hog_vis(img, orientations=8, pixels_per_cell=8, cells_per_block=2):
    from skimage.feature import hog
    gray = _float(_gray(img))
    _, hog_img = hog(gray, orientations=orientations,
                      pixels_per_cell=(pixels_per_cell, pixels_per_cell),
                      cells_per_block=(cells_per_block, cells_per_block),
                      visualize=True)
    return _norm8(hog_img)

@_try_skimage
def _daisy_vis(img, step=4, radius=15, rings=2, histograms=6, orientations=8):
    from skimage.feature import daisy
    gray = _float(_gray(img))
    descs, descs_img = daisy(gray, step=step, radius=radius, rings=rings,
                               histograms=histograms, orientations=orientations,
                               visualize=True)
    return _norm8(descs_img)

@_try_skimage
def _glcm_features(img) -> dict:
    from skimage.feature import graycomatrix, graycoprops
    gray = (_gray(img) // 4).astype(np.uint8)  # quantize to 64 levels
    glcm = graycomatrix(gray, distances=[1,2], angles=[0, np.pi/4, np.pi/2],
                         levels=64, symmetric=True, normed=True)
    result = {}
    for prop in ['contrast','dissimilarity','homogeneity','energy','correlation','ASM']:
        vals = graycoprops(glcm, prop).flatten()
        for i, v in enumerate(vals): result[f"glcm_{prop}_{i}"] = float(v)
    return result

@_try_skimage
def _region_props(img) -> dict:
    from skimage.measure import label, regionprops
    gray = _gray(img)
    binary = gray > 127
    labeled = label(binary)
    regions = regionprops(labeled)
    if not regions:
        return {}
    r = max(regions, key=lambda x: x.area)
    return {
        "area": float(r.area),
        "perimeter": float(r.perimeter),
        "eccentricity": float(r.eccentricity),
        "solidity": float(r.solidity),
        "extent": float(r.extent),
        "major_axis": float(r.major_axis_length),
        "minor_axis": float(r.minor_axis_length),
        "orientation": float(r.orientation),
        "euler_number": float(r.euler_number),
    }


# ═══ IMAGE QUALITY METRICS ═════════════════════════════════════════════════════

def _ssim_metric(img1, img2) -> dict:
    try:
        from skimage.metrics import structural_similarity as ssim
        from skimage.metrics import peak_signal_noise_ratio, mean_squared_error
        g1 = _gray(img1).astype(np.float64)
        g2 = _gray(img2).astype(np.float64)
        if g1.shape != g2.shape:
            g2 = cv2.resize(g2.astype(np.uint8), (g1.shape[1], g1.shape[0])).astype(np.float64)
        ssim_val = ssim(g1, g2, data_range=255.0)
        psnr_val = peak_signal_noise_ratio(g1, g2, data_range=255.0)
        mse_val = mean_squared_error(g1, g2)
        rmse_val = float(np.sqrt(mse_val))
        return {"SSIM": round(ssim_val,4), "PSNR_dB": round(psnr_val,2), "RMSE": round(rmse_val,2), "MSE": round(mse_val,2)}
    except ImportError:
        return {"error": "scikit-image not installed"}


# ═══ TEMPLATE MATCHING ════════════════════════════════════════════════════════

@_try_skimage
def _template_match(img):
    from skimage.feature import match_template
    gray = _float(_gray(img))
    h, w = gray.shape
    # Use top-left 1/4 as template
    template = gray[:h//4, :w//4]
    result = match_template(gray, template)
    out = _bgr(img).copy()
    ij = np.unravel_index(np.argmax(result), result.shape)
    y, x = ij
    cv2.rectangle(out, (x, y), (x + w//4, y + h//4), (0, 255, 0), 2)
    cv2.rectangle(out, (0, 0), (w//4, h//4), (0, 0, 255), 2)
    return out


# ═══ REGISTER ALL ══════════════════════════════════════════════════════════════

def register_all():
    specs = [
        # Segmentation
        AlgorithmSpec("ski_slic","SLIC Superpixels",_L,"Analysis","Segmentation",_slic,
            [Param.Int("n_segments","Segments",100,10,500,10),Param.Float("compactness","Compactness",10.0,0.1,50.0,1.0)],_I,"Simple Linear Iterative Clustering"),
        AlgorithmSpec("ski_felzenszwalb","Felzenszwalb",_L,"Analysis","Segmentation",_felzenszwalb,
            [Param.Float("scale","Scale",100.0,10.0,500.0,10.0),Param.Float("sigma","Sigma",0.5,0.1,2.0,0.1),Param.Int("min_size","Min size",50,10,500,10)],_I),
        AlgorithmSpec("ski_quickshift","Quickshift",_L,"Analysis","Segmentation",_quickshift,
            [Param.Int("kernel_size","Kernel size",3,1,10,1),Param.Float("max_dist","Max dist",6.0,1.0,20.0,0.5)],_I),
        AlgorithmSpec("ski_chan_vese","Chan-Vese",_L,"Analysis","Segmentation",_chan_vese,
            [Param.Float("mu","mu",0.25,0.01,1.0,0.01),Param.Int("max_iter","Max iter",200,10,1000,10)],_I,"Active contour without edges"),
        AlgorithmSpec("ski_active_contour","Active Contour",_L,"Analysis","Segmentation",_active_contour,
            [Param.Float("alpha","Alpha",0.015,0.001,0.1,0.001),Param.Float("beta","Beta",10.0,1.0,100.0,1.0)],_O,"Snake active contour model"),

        # Edge Detection
        AlgorithmSpec("ski_log","LoG (Marr-Hildreth)",_L,"Analysis","Edge Detection",_log_edge,
            [Param.Float("sigma","Sigma",2.0,0.5,10.0,0.5)],_I,"Laplacian of Gaussian"),
        AlgorithmSpec("ski_frangi","Frangi Vessel",_L,"Analysis","Edge Detection",_frangi,
            [Param.Int("sigmas_start","σ start",1,1,5,1),Param.Int("sigmas_stop","σ stop",10,5,20,1)],_I,"Frangi vesselness filter"),
        AlgorithmSpec("ski_sato","Sato Tubular",_L,"Analysis","Edge Detection",_sato,[],_I,"Sato tubular structure filter"),
        AlgorithmSpec("ski_meijering","Meijering Neurite",_L,"Analysis","Edge Detection",_meijering,[],_I,"Meijering neuriteness filter"),

        # Restoration
        AlgorithmSpec("ski_wavelet_denoise","Wavelet Denoise",_L,"Restoration","Denoising",_wavelet_denoise,
            [Param.Choice("wavelet","Wavelet",['db1','db2','haar','sym2'])],_I,"scikit-image wavelet denoising"),
        AlgorithmSpec("ski_tv_denoise","Total Variation Denoise",_L,"Restoration","Denoising",_tv_denoise,
            [Param.Float("weight","Weight",0.1,0.001,1.0,0.01)],_I,"Chambolle TV denoising"),
        AlgorithmSpec("ski_inpaint","Inpaint Biharmonic",_L,"Restoration","Inpainting",_inpaint_biharmonic,
            [Param.Int("radius","Mask radius",5,1,30,1)],_I,"Biharmonic inpainting on dark regions"),
        AlgorithmSpec("ski_rl","Richardson-Lucy Deconvolve",_L,"Restoration","Deconvolution",_richardson_lucy,
            [Param.Int("psf_size","PSF size",5,3,15,2),Param.Int("iterations","Iterations",10,1,50,1)],_I),

        # Enhancement
        AlgorithmSpec("ski_adapt_hist","Adaptive Hist Equalize",_L,"Enhancement","Histogram",_equalize_adapthist,
            [Param.Float("clip_limit","Clip limit",0.01,0.001,0.1,0.001)],_I),
        AlgorithmSpec("ski_gamma","Gamma Correction",_L,"Enhancement","Tonal",_adjust_gamma,
            [Param.Float("gamma","Gamma",1.0,0.1,5.0,0.05)],_I),
        AlgorithmSpec("ski_sigmoid","Sigmoid Contrast",_L,"Enhancement","Tonal",_adjust_sigmoid,
            [Param.Float("cutoff","Cutoff",0.5,0.0,1.0,0.05),Param.Float("gain","Gain",10.0,1.0,50.0,1.0)],_I),
        AlgorithmSpec("ski_rescale","Intensity Rescale",_L,"Enhancement","Tonal",_rescale_intensity,
            [Param.Float("plo","Low %",2.0,0.0,20.0,0.5),Param.Float("phi","High %",98.0,80.0,100.0,0.5)],_I),

        # Morphology extras
        AlgorithmSpec("ski_bin_close","Binary Closing",_L,"Analysis","Morphology",_binary_closing,
            [Param.Int("radius","Disk radius",3,1,20,1)],_I),
        AlgorithmSpec("ski_rm_small","Remove Small Objects",_L,"Analysis","Morphology",_remove_small_objects,
            [Param.Int("min_size","Min size px",64,4,1000,4)],_I),
        AlgorithmSpec("ski_convex","Convex Hull",_L,"Analysis","Morphology",_convex_hull,[],_O),

        # Texture & Features
        AlgorithmSpec("ski_lbp","Local Binary Pattern",_L,"Analysis","Texture",_local_binary_pattern,
            [Param.Int("P","Points P",8,4,24,4),Param.Float("R","Radius R",1.0,1.0,5.0,0.5)],_I,"LBP texture descriptor"),
        AlgorithmSpec("ski_hog","HOG (visualised)",_L,"Analysis","Texture",_hog_vis,
            [Param.Int("orientations","Orientations",8,4,16,4),Param.Int("pixels_per_cell","Px/cell",8,4,32,4)],_I,"Histogram of Oriented Gradients"),
        AlgorithmSpec("ski_daisy","DAISY (visualised)",_L,"Analysis","Texture",_daisy_vis,[],_I,"DAISY descriptor visualisation"),

        # Feature extraction (returns dict)
        AlgorithmSpec("ski_glcm","GLCM Texture Features",_L,"Analysis","Feature Extraction",_glcm_features,[],_F,"Gray-Level Co-occurrence Matrix features"),
        AlgorithmSpec("ski_regionprops","Region Properties",_L,"Analysis","Feature Extraction",_region_props,[],_F,"scikit-image region properties"),

        # Template matching
        AlgorithmSpec("ski_template","Template Matching",_L,"Analysis","Feature Detection",_template_match,[],_O,"Match top-left quadrant as template"),
    ]
    for s in specs:
        REGISTRY.register(s)
