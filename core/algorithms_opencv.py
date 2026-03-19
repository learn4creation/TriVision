"""
TriVision — OpenCV Algorithm Registrations
All algorithms sourced from cv2.
"""

import cv2
import numpy as np
from .registry import REGISTRY, AlgorithmSpec, Param, Lib, ReturnType, register

_L = Lib.OPENCV
_I = ReturnType.IMAGE
_F = ReturnType.FEATURES
_O = ReturnType.OVERLAY


def _gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img

def _norm8(a):
    a = a.astype(np.float64)
    mn, mx = a.min(), a.max()
    return ((a - mn) / (mx - mn + 1e-10) * 255).astype(np.uint8)

def _odd(k): return k if k % 2 == 1 else k + 1

def _bgr(img):
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if len(img.shape) == 2 else img


# ═══ EDGE DETECTION ═══════════════════════════════════════════════════════════

def _sobel(img):
    g = _gray(img).astype(np.float32)
    return _norm8(np.sqrt(cv2.Sobel(g,cv2.CV_32F,1,0)**2 + cv2.Sobel(g,cv2.CV_32F,0,1)**2))

def _prewitt(img):
    g = _gray(img).astype(np.float32)
    kx = np.array([[-1,0,1],[-1,0,1],[-1,0,1]], np.float32)
    return _norm8(np.sqrt(cv2.filter2D(g,-1,kx)**2 + cv2.filter2D(g,-1,kx.T)**2))

def _roberts(img):
    g = _gray(img).astype(np.float32)
    kx = np.array([[1,0],[0,-1]], np.float32)
    ky = np.array([[0,1],[-1,0]], np.float32)
    return _norm8(np.sqrt(cv2.filter2D(g,-1,kx)**2 + cv2.filter2D(g,-1,ky)**2))

def _laplacian(img, ksize=3):
    return _norm8(np.abs(cv2.Laplacian(_gray(img), cv2.CV_64F, ksize=_odd(ksize))))

def _canny(img, low=50, high=150):
    return cv2.Canny(_gray(img), low, high)

def _kirsch(img):
    g = _gray(img).astype(np.float32)
    s2 = np.sqrt(2)
    masks = [np.array([[5,5,5],[-3,0,-3],[-3,-3,-3]],np.float32),
             np.array([[-3,5,5],[-3,0,5],[-3,-3,-3]],np.float32),
             np.array([[-3,-3,5],[-3,0,5],[-3,-3,5]],np.float32),
             np.array([[-3,-3,-3],[-3,0,5],[-3,5,5]],np.float32),
             np.array([[-3,-3,-3],[-3,0,-3],[5,5,5]],np.float32),
             np.array([[-3,-3,-3],[5,0,-3],[5,5,-3]],np.float32),
             np.array([[5,-3,-3],[5,0,-3],[5,-3,-3]],np.float32),
             np.array([[5,5,-3],[5,0,-3],[-3,-3,-3]],np.float32)]
    return _norm8(np.max(np.stack([np.abs(cv2.filter2D(g,-1,m)) for m in masks]),axis=0))

def _hough_lines(img, threshold=80):
    edges = cv2.Canny(_gray(img), 50, 150)
    result = _bgr(img.copy())
    lines = cv2.HoughLines(edges, 1, np.pi/180, threshold)
    if lines is not None:
        for rho, theta in lines[:min(50,len(lines)),0]:
            a,b = np.cos(theta), np.sin(theta)
            cv2.line(result,(int(a*rho+1000*(-b)),int(b*rho+1000*a)),
                             (int(a*rho-1000*(-b)),int(b*rho-1000*a)),(0,0,255),1)
    return result

def _hough_circles(img, min_r=10, max_r=100):
    gray = _gray(img)
    result = _bgr(img.copy())
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1.2, 30,
                                param1=100, param2=30, minRadius=min_r, maxRadius=max_r)
    if circles is not None:
        for x,y,r in np.uint16(np.around(circles[0])):
            cv2.circle(result,(x,y),r,(0,255,0),2)
            cv2.circle(result,(x,y),2,(0,0,255),3)
    return result


# ═══ SEGMENTATION ══════════════════════════════════════════════════════════════

def _thresh(img, low=127, high=255):
    _,out = cv2.threshold(_gray(img), low, high, cv2.THRESH_BINARY)
    return out

def _otsu(img):
    _,out = cv2.threshold(_gray(img), 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    return out

def _adaptive(img, block_size=11, C=2):
    return cv2.adaptiveThreshold(_gray(img),255,
           cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,_odd(block_size),C)

def _kmeans_seg(img, k=4):
    data = img.reshape((-1,img.shape[2] if len(img.shape)==3 else 1)).astype(np.float32)
    crit = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER,20,0.5)
    _,labels,centers = cv2.kmeans(data,k,None,crit,5,cv2.KMEANS_RANDOM_CENTERS)
    return np.uint8(centers)[labels.flatten()].reshape(img.shape)

def _watershed_seg(img):
    gray = _gray(img)
    _,binary = cv2.threshold(gray,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    k = np.ones((3,3),np.uint8)
    opening = cv2.morphologyEx(binary,cv2.MORPH_OPEN,k,iterations=2)
    sure_bg = cv2.dilate(opening,k,iterations=3)
    dist = cv2.distanceTransform(opening,cv2.DIST_L2,5)
    _,sure_fg = cv2.threshold(dist,0.5*dist.max(),255,0)
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg,sure_fg)
    _,markers = cv2.connectedComponents(sure_fg)
    markers += 1; markers[unknown==255] = 0
    result = _bgr(img) if len(img.shape)==2 else img.copy()
    cv2.watershed(result,markers)
    result[markers==-1] = [0,0,255]
    return result

def _grabcut(img, iter=5):
    h,w = img.shape[:2]
    mask = np.zeros((h,w),np.uint8)
    bgd = np.zeros((1,65),np.float64)
    fgd = np.zeros((1,65),np.float64)
    rect = (w//8,h//8,w*3//4,h*3//4)
    cv2.grabCut(_bgr(img),mask,rect,bgd,fgd,iter,cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask==2)|(mask==0),0,1).astype(np.uint8)
    return (_bgr(img)*mask2[:,:,np.newaxis]).astype(np.uint8)


# ═══ MORPHOLOGY ════════════════════════════════════════════════════════════════

def _k(s): return np.ones((_odd(s),_odd(s)),np.uint8)

def _erode(img,ksize=5,iters=1):   return cv2.erode(img,_k(ksize),iterations=iters)
def _dilate(img,ksize=5,iters=1):  return cv2.dilate(img,_k(ksize),iterations=iters)
def _morph_open(img,ksize=5):      return cv2.morphologyEx(img,cv2.MORPH_OPEN,_k(ksize))
def _morph_close(img,ksize=5):     return cv2.morphologyEx(img,cv2.MORPH_CLOSE,_k(ksize))
def _morph_gradient(img,ksize=5):  return cv2.morphologyEx(img,cv2.MORPH_GRADIENT,_k(ksize))
def _tophat(img,ksize=15):         return cv2.morphologyEx(img,cv2.MORPH_TOPHAT,_k(ksize))
def _blackhat(img,ksize=15):       return cv2.morphologyEx(img,cv2.MORPH_BLACKHAT,_k(ksize))

def _skeleton(img):
    _,binary = cv2.threshold(_gray(img),127,255,cv2.THRESH_BINARY)
    skel = np.zeros_like(binary)
    el = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
    tmp = binary.copy()
    while True:
        eroded = cv2.erode(tmp,el)
        skel |= cv2.subtract(tmp,cv2.dilate(eroded,el))
        tmp = eroded
        if not cv2.countNonZero(tmp): break
    return skel

def _connected_label(img):
    _,binary = cv2.threshold(_gray(img),127,255,cv2.THRESH_BINARY)
    n,labels = cv2.connectedComponents(binary)
    colors = np.random.RandomState(42).randint(40,255,(n,3),dtype=np.uint8)
    colors[0] = [0,0,0]
    return colors[labels]


# ═══ TRANSFORMS ════════════════════════════════════════════════════════════════

def _dft_mag(img):
    gray = _gray(img).astype(np.float32)
    dft = np.fft.fftshift(np.fft.fft2(gray))
    return _norm8(np.log1p(np.abs(dft)))

def _dft_phase(img):
    gray = _gray(img).astype(np.float32)
    dft = np.fft.fftshift(np.fft.fft2(gray))
    return _norm8(np.angle(dft)+np.pi)

def _dct_img(img):
    return _norm8(np.log1p(np.abs(cv2.dct(_gray(img).astype(np.float32)))))

def _haar(img):
    gray = _gray(img).astype(np.float32)
    h,w = gray.shape; gray = gray[:h&~1,:w&~1]
    L=(gray[:,::2]+gray[:,1::2])/2; Hi=(gray[:,::2]-gray[:,1::2])/2
    LL=(L[::2,:]+L[1::2,:])/2; LH=(L[::2,:]-L[1::2,:])/2
    HL=(Hi[::2,:]+Hi[1::2,:])/2; HH=(Hi[::2,:]-Hi[1::2,:])/2
    return np.vstack([np.hstack([_norm8(LL),_norm8(LH)]),
                      np.hstack([_norm8(HL),_norm8(HH)])])


# ═══ FEATURE DETECTION ═════════════════════════════════════════════════════════

def _orb(img, max_kp=500):
    kps = cv2.ORB_create(max_kp).detect(_gray(img),None)
    return cv2.drawKeypoints(_bgr(img),kps,None,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

def _akaze(img):
    kps = cv2.AKAZE_create().detect(_gray(img),None)
    return cv2.drawKeypoints(_bgr(img),kps,None,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

def _brisk(img):
    kps = cv2.BRISK_create().detect(_gray(img),None)
    return cv2.drawKeypoints(_bgr(img),kps,None,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

def _harris(img):
    gray = np.float32(_gray(img))
    h = cv2.dilate(cv2.cornerHarris(gray,2,3,0.04),None)
    res = _bgr(img).copy(); res[h>0.01*h.max()]=[0,0,255]
    return res

def _shi_tomasi(img, max_corners=300):
    corners = cv2.goodFeaturesToTrack(_gray(img),max_corners,0.01,10)
    res = _bgr(img).copy()
    if corners is not None:
        for c in np.intp(corners): cv2.circle(res,tuple(c.ravel()),3,(0,255,0),-1)
    return res

def _fast(img):
    kps = cv2.FastFeatureDetector_create().detect(_gray(img),None)
    return cv2.drawKeypoints(_bgr(img),kps,None,color=(255,0,0))

def _sift(img):
    try:
        kps,_ = cv2.SIFT_create().detectAndCompute(_gray(img),None)
        return cv2.drawKeypoints(_bgr(img),kps,None,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    except Exception:
        return _bgr(img)


# ═══ ENHANCEMENT ═══════════════════════════════════════════════════════════════

def _hist_eq(img):
    if len(img.shape)==3:
        y = cv2.cvtColor(img,cv2.COLOR_BGR2YCrCb)
        y[:,:,0] = cv2.equalizeHist(y[:,:,0])
        return cv2.cvtColor(y,cv2.COLOR_YCrCb2BGR)
    return cv2.equalizeHist(img)

def _clahe(img, clip=2.0, tile=8):
    c = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile,tile))
    if len(img.shape)==3:
        y = cv2.cvtColor(img,cv2.COLOR_BGR2YCrCb)
        y[:,:,0] = c.apply(y[:,:,0])
        return cv2.cvtColor(y,cv2.COLOR_YCrCb2BGR)
    return c.apply(img)

def _unsharp(img, sigma=1.0, strength=1.5):
    blurred = cv2.GaussianBlur(img,(0,0),sigma)
    return np.clip(cv2.addWeighted(img,1+strength,blurred,-strength,0),0,255).astype(np.uint8)

def _bilateral(img, d=9, sigma_color=75, sigma_space=75):
    return cv2.bilateralFilter(img,d,sigma_color,sigma_space)

def _denoising(img, h=10):
    return cv2.fastNlMeansDenoisingColored(img,None,h,h,7,21) if len(img.shape)==3 \
           else cv2.fastNlMeansDenoising(img,None,h,7,21)

def _detail_enhance(img, sigma_s=10, sigma_r=0.15):
    return cv2.detailEnhance(_bgr(img),sigma_s=sigma_s,sigma_r=sigma_r)

def _stylize(img, sigma_s=60, sigma_r=0.45):
    return cv2.stylization(_bgr(img),sigma_s=sigma_s,sigma_r=sigma_r)

def _pencil_sketch(img):
    _,color = cv2.pencilSketch(_bgr(img),sigma_s=60,sigma_r=0.07,shade_factor=0.05)
    return color

def _colormap(cmap):
    def fn(img): return cv2.applyColorMap(_gray(img), cmap)
    return fn


# ═══ RESTORATION ═══════════════════════════════════════════════════════════════

def _noise_gaussian(img, mean=0.0, var=200.0):
    return np.clip(img.astype(np.float32)+np.random.normal(mean,var**0.5,img.shape),0,255).astype(np.uint8)

def _noise_sp(img, density=0.05):
    out = img.copy()
    n = int(img.size//(img.shape[2] if len(img.shape)==3 else 1) * density/2)
    for val in [255,0]:
        coords = [np.random.randint(0,d,n) for d in img.shape[:2]]
        out[coords[0],coords[1]] = val
    return out

def _geom_rotate(img, angle=15.0):
    h,w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w/2,h/2),angle,1.0)
    return cv2.warpAffine(img,M,(w,h),flags=cv2.INTER_LINEAR)

def _geom_warp(img, amplitude=20.0):
    h,w = img.shape[:2]
    mx = np.zeros((h,w),np.float32); my = np.zeros((h,w),np.float32)
    for r in range(h):
        for c in range(w):
            mx[r,c]=c+amplitude*np.sin(2*np.pi*r/max(h,1))
            my[r,c]=r+amplitude*np.sin(2*np.pi*c/max(w,1))
    return cv2.remap(img,mx,my,cv2.INTER_LINEAR)

def _barrel(img, k=-0.3):
    h,w = img.shape[:2]
    K = np.array([[w,0,w/2],[0,h,h/2],[0,0,1]],np.float64)
    return cv2.undistort(img,K,np.array([k,k*0.5,0,0,0]))


# ═══ COMPRESSION ═══════════════════════════════════════════════════════════════

def _jpeg(img, quality=20):
    s,buf = cv2.imencode('.jpg',img,[cv2.IMWRITE_JPEG_QUALITY,quality])
    if not s: return img,1.0,8.0
    out = cv2.imdecode(buf, cv2.IMREAD_COLOR if len(img.shape)==3 else cv2.IMREAD_GRAYSCALE)
    r = img.size/len(buf); bpp = 8/r
    return out, r, bpp

def _webp(img, quality=30):
    s,buf = cv2.imencode('.webp',img,[cv2.IMWRITE_WEBP_QUALITY,quality])
    if not s: return img,1.0,8.0
    out = cv2.imdecode(buf, cv2.IMREAD_COLOR if len(img.shape)==3 else cv2.IMREAD_GRAYSCALE)
    r = img.size/len(buf); bpp = 8/r
    return out, r, bpp


# ═══ OBJECT DETECTION ══════════════════════════════════════════════════════════

def _face_haar(img):
    res = _bgr(img).copy()
    gray = _gray(img)
    cc = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
    for x,y,w,h in cc.detectMultiScale(gray,1.1,4):
        cv2.rectangle(res,(x,y),(x+w,y+h),(0,255,0),2)
        cv2.putText(res,'Face',(x,y-4),cv2.FONT_HERSHEY_SIMPLEX,0.5,(0,255,0),1)
    return res

def _eye_haar(img):
    res = _bgr(img).copy()
    cc = cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_eye.xml')
    for x,y,w,h in cc.detectMultiScale(_gray(img),1.1,4):
        cv2.rectangle(res,(x,y),(x+w,y+h),(255,0,0),2)
    return res

def _qr_detect(img):
    res = _bgr(img).copy()
    data,bbox,_ = cv2.QRCodeDetector().detectAndDecode(res)
    if bbox is not None:
        cv2.polylines(res,[np.intp(bbox[0])],True,(0,255,0),2)
        if data: cv2.putText(res,data,(10,30),cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0),2)
    return res


# ═══ REGISTER ALL ══════════════════════════════════════════════════════════════

def register_all():
    specs = [
        # ── Edge Detection ────────────────────────────────────────────
        AlgorithmSpec("ocv_sobel","Sobel",_L,"Analysis","Edge Detection",_sobel,[],_I,"Gradient magnitude via Sobel operators"),
        AlgorithmSpec("ocv_prewitt","Prewitt",_L,"Analysis","Edge Detection",_prewitt,[],_I,"Prewitt compass gradient"),
        AlgorithmSpec("ocv_roberts","Roberts",_L,"Analysis","Edge Detection",_roberts,[],_I,"Roberts cross gradient"),
        AlgorithmSpec("ocv_laplacian","Laplacian",_L,"Analysis","Edge Detection",_laplacian,
            [Param.Int("ksize","Kernel",3,1,15,2)],_I,"Second-order Laplacian edges"),
        AlgorithmSpec("ocv_canny","Canny",_L,"Analysis","Edge Detection",_canny,
            [Param.Int("low","Low thr",50,0,255,5),Param.Int("high","High thr",150,0,255,5)],_I,"Canny multi-stage edge detector"),
        AlgorithmSpec("ocv_kirsch","Kirsch",_L,"Analysis","Edge Detection",_kirsch,[],_I,"Kirsch 8-direction compass masks"),
        AlgorithmSpec("ocv_hough_lines","Hough Lines",_L,"Analysis","Edge Detection",_hough_lines,
            [Param.Int("threshold","Threshold",80,10,300,10)],_O,"Hough transform line detection"),
        AlgorithmSpec("ocv_hough_circles","Hough Circles",_L,"Analysis","Edge Detection",_hough_circles,
            [Param.Int("min_r","Min radius",10,1,100,5),Param.Int("max_r","Max radius",100,20,400,10)],_O,"Hough circle detection"),

        # ── Segmentation ──────────────────────────────────────────────
        AlgorithmSpec("ocv_thresh","Threshold",_L,"Analysis","Segmentation",_thresh,
            [Param.Int("low","Low",127,0,254,1),Param.Int("high","High",255,1,255,1)],_I),
        AlgorithmSpec("ocv_otsu","Otsu",_L,"Analysis","Segmentation",_otsu,[],_I,"Automatic optimal threshold"),
        AlgorithmSpec("ocv_adaptive","Adaptive Threshold",_L,"Analysis","Segmentation",_adaptive,
            [Param.Int("block_size","Block size",11,3,51,2),Param.Int("C","C offset",2,-20,20,1)],_I),
        AlgorithmSpec("ocv_kmeans","K-Means",_L,"Analysis","Segmentation",_kmeans_seg,
            [Param.Int("k","K clusters",4,2,16,1)],_I,"K-means color segmentation"),
        AlgorithmSpec("ocv_watershed","Watershed",_L,"Analysis","Segmentation",_watershed_seg,[],_O,"Marker-based watershed"),
        AlgorithmSpec("ocv_grabcut","GrabCut",_L,"Analysis","Segmentation",_grabcut,
            [Param.Int("iter","Iterations",5,1,20,1)],_I,"Interactive GrabCut foreground extraction"),

        # ── Morphology ────────────────────────────────────────────────
        AlgorithmSpec("ocv_erode","Erode",_L,"Analysis","Morphology",_erode,
            [Param.Int("ksize","Kernel",5,1,31,2),Param.Int("iters","Iterations",1,1,10,1)],_I),
        AlgorithmSpec("ocv_dilate","Dilate",_L,"Analysis","Morphology",_dilate,
            [Param.Int("ksize","Kernel",5,1,31,2),Param.Int("iters","Iterations",1,1,10,1)],_I),
        AlgorithmSpec("ocv_open","Morph Open",_L,"Analysis","Morphology",_morph_open,
            [Param.Int("ksize","Kernel",5,1,31,2)],_I),
        AlgorithmSpec("ocv_close","Morph Close",_L,"Analysis","Morphology",_morph_close,
            [Param.Int("ksize","Kernel",5,1,31,2)],_I),
        AlgorithmSpec("ocv_gradient","Morph Gradient",_L,"Analysis","Morphology",_morph_gradient,
            [Param.Int("ksize","Kernel",5,1,31,2)],_I),
        AlgorithmSpec("ocv_tophat","Top Hat",_L,"Analysis","Morphology",_tophat,
            [Param.Int("ksize","Kernel",15,1,61,2)],_I),
        AlgorithmSpec("ocv_blackhat","Black Hat",_L,"Analysis","Morphology",_blackhat,
            [Param.Int("ksize","Kernel",15,1,61,2)],_I),
        AlgorithmSpec("ocv_skeleton","Skeleton",_L,"Analysis","Morphology",_skeleton,[],_I,"Morphological skeleton via iterative erosion"),
        AlgorithmSpec("ocv_label","Connected Labels",_L,"Analysis","Morphology",_connected_label,[],_I,"Connected component labeling"),

        # ── Transforms ───────────────────────────────────────────────
        AlgorithmSpec("ocv_dft_mag","DFT Magnitude",_L,"Analysis","Transforms",_dft_mag,[],_I,"Log-magnitude Fourier spectrum"),
        AlgorithmSpec("ocv_dft_phase","DFT Phase",_L,"Analysis","Transforms",_dft_phase,[],_I,"Fourier phase spectrum"),
        AlgorithmSpec("ocv_dct","DCT",_L,"Analysis","Transforms",_dct_img,[],_I,"Discrete Cosine Transform"),
        AlgorithmSpec("ocv_haar","Haar Wavelet",_L,"Analysis","Transforms",_haar,[],_I,"1-level Haar wavelet 4-subband"),

        # ── Feature Detection ────────────────────────────────────────
        AlgorithmSpec("ocv_orb","ORB",_L,"Analysis","Feature Detection",_orb,
            [Param.Int("max_kp","Max keypoints",500,10,2000,50)],_O,"ORB keypoints (rotation invariant)"),
        AlgorithmSpec("ocv_akaze","AKAZE",_L,"Analysis","Feature Detection",_akaze,[],_O,"AKAZE keypoints"),
        AlgorithmSpec("ocv_brisk","BRISK",_L,"Analysis","Feature Detection",_brisk,[],_O,"BRISK keypoints"),
        AlgorithmSpec("ocv_harris","Harris Corners",_L,"Analysis","Feature Detection",_harris,[],_O,"Harris corner detector"),
        AlgorithmSpec("ocv_shi_tomasi","Shi-Tomasi",_L,"Analysis","Feature Detection",_shi_tomasi,
            [Param.Int("max_corners","Max corners",300,10,2000,50)],_O,"Good features to track"),
        AlgorithmSpec("ocv_fast","FAST",_L,"Analysis","Feature Detection",_fast,[],_O,"FAST corner detector"),
        AlgorithmSpec("ocv_sift","SIFT",_L,"Analysis","Feature Detection",_sift,[],_O,"Scale-Invariant Feature Transform"),

        # ── Enhancement ──────────────────────────────────────────────
        AlgorithmSpec("ocv_hist_eq","Histogram Equalize",_L,"Enhancement","Histogram",_hist_eq,[],_I),
        AlgorithmSpec("ocv_clahe","CLAHE",_L,"Enhancement","Histogram",_clahe,
            [Param.Float("clip","Clip limit",2.0,0.1,20.0,0.5),Param.Int("tile","Tile size",8,2,32,2)],_I,"Contrast Limited AHE"),
        AlgorithmSpec("ocv_unsharp","Unsharp Mask",_L,"Enhancement","Spatial",_unsharp,
            [Param.Float("sigma","Sigma",1.0,0.1,10.0,0.1),Param.Float("strength","Strength",1.5,0.0,5.0,0.1)],_I),
        AlgorithmSpec("ocv_bilateral","Bilateral Filter",_L,"Enhancement","Spatial",_bilateral,
            [Param.Int("d","Diameter",9,1,25,2),Param.Float("sigma_color","σ color",75.0,1.0,200.0,5.0),Param.Float("sigma_space","σ space",75.0,1.0,200.0,5.0)],_I),
        AlgorithmSpec("ocv_nlm","NL Means Denoise",_L,"Enhancement","Spatial",_denoising,
            [Param.Float("h","Filter strength",10.0,1.0,50.0,1.0)],_I),
        AlgorithmSpec("ocv_detail","Detail Enhance",_L,"Enhancement","Computational Photo",_detail_enhance,
            [Param.Float("sigma_s","Sigma S",10.0,1.0,200.0,1.0),Param.Float("sigma_r","Sigma R",0.15,0.01,1.0,0.01)],_I),
        AlgorithmSpec("ocv_stylize","Stylization",_L,"Enhancement","Computational Photo",_stylize,
            [Param.Float("sigma_s","Sigma S",60.0,1.0,200.0,1.0),Param.Float("sigma_r","Sigma R",0.45,0.01,1.0,0.01)],_I),
        AlgorithmSpec("ocv_pencil","Pencil Sketch",_L,"Enhancement","Computational Photo",_pencil_sketch,[],_I),
        AlgorithmSpec("ocv_cmap_jet","Colormap JET",_L,"Enhancement","Pseudo-Color",_colormap(cv2.COLORMAP_JET),[],_I),
        AlgorithmSpec("ocv_cmap_hot","Colormap HOT",_L,"Enhancement","Pseudo-Color",_colormap(cv2.COLORMAP_HOT),[],_I),
        AlgorithmSpec("ocv_cmap_viridis","Colormap VIRIDIS",_L,"Enhancement","Pseudo-Color",_colormap(cv2.COLORMAP_VIRIDIS),[],_I),
        AlgorithmSpec("ocv_cmap_turbo","Colormap TURBO",_L,"Enhancement","Pseudo-Color",_colormap(cv2.COLORMAP_TURBO),[],_I),
        AlgorithmSpec("ocv_cmap_magma","Colormap MAGMA",_L,"Enhancement","Pseudo-Color",_colormap(cv2.COLORMAP_MAGMA),[],_I),

        # ── Noise ────────────────────────────────────────────────────
        AlgorithmSpec("ocv_noise_gauss","Gaussian Noise",_L,"Restoration","Noise Models",_noise_gaussian,
            [Param.Float("mean","Mean",0.0,-50.0,50.0,1.0),Param.Float("var","Variance",200.0,1.0,2000.0,10.0)],_I),
        AlgorithmSpec("ocv_noise_sp","Salt & Pepper Noise",_L,"Restoration","Noise Models",_noise_sp,
            [Param.Float("density","Density",0.05,0.001,0.3,0.005)],_I),

        # ── Geometric ────────────────────────────────────────────────
        AlgorithmSpec("ocv_rotate","Rotate",_L,"Restoration","Geometric",_geom_rotate,
            [Param.Float("angle","Angle °",-15.0,-180.0,180.0,1.0)],_I),
        AlgorithmSpec("ocv_warp","Sinusoidal Warp",_L,"Restoration","Geometric",_geom_warp,
            [Param.Float("amplitude","Amplitude",20.0,0.0,80.0,1.0)],_I),
        AlgorithmSpec("ocv_barrel","Barrel Distortion",_L,"Restoration","Geometric",_barrel,
            [Param.Float("k","k",-0.3,-1.0,1.0,0.05)],_I),

        # ── Compression ──────────────────────────────────────────────
        AlgorithmSpec("ocv_jpeg","JPEG",_L,"Compression","Lossy",_jpeg,
            [Param.Int("quality","Quality",20,1,100,1)],ReturnType.TUPLE,"JPEG compression via OpenCV"),
        AlgorithmSpec("ocv_webp","WebP",_L,"Compression","Lossy",_webp,
            [Param.Int("quality","Quality",30,1,100,1)],ReturnType.TUPLE,"WebP compression via OpenCV"),

        # ── Object Detection ─────────────────────────────────────────
        AlgorithmSpec("ocv_face","Face Detection (Haar)",_L,"Analysis","Object Detection",_face_haar,[],_O),
        AlgorithmSpec("ocv_eye","Eye Detection (Haar)",_L,"Analysis","Object Detection",_eye_haar,[],_O),
        AlgorithmSpec("ocv_qr","QR Code Detection",_L,"Analysis","Object Detection",_qr_detect,[],_O),
    ]
    for s in specs:
        REGISTRY.register(s)
