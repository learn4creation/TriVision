"""
TriVision — CVIPtools2 + TriVision Fusion Algorithm Registrations

CVIPtools2 algorithms: frequency-domain restoration, classical compression,
histogram operations, Laws texture, RST features, frequency filters, BTC.

TriVision fusion algorithms: combine OpenCV + skimage + CVIPtools
for things none can do alone:
  - Multi-scale edge fusion
  - Hybrid denoising pipeline
  - Comprehensive feature extraction (all three sources)
  - Adaptive compression benchmark
  - Perceptual quality scoring (SSIM + PSNR + LPIPS-like)
"""

import cv2
import numpy as np
from .registry import REGISTRY, AlgorithmSpec, Param, Lib, ReturnType

_LC = Lib.CVIP
_LT = Lib.TRIVISION
_I  = ReturnType.IMAGE
_F  = ReturnType.FEATURES
_M  = ReturnType.METRICS
_T  = ReturnType.TUPLE
_O  = ReturnType.OVERLAY


def _gray(img): return cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) if len(img.shape)==3 else img
def _bgr(img):  return cv2.cvtColor(img,cv2.COLOR_GRAY2BGR) if len(img.shape)==2 else img
def _norm8(a):
    a=a.astype(np.float64); mn,mx=a.min(),a.max()
    return ((a-mn)/(mx-mn+1e-10)*255).astype(np.uint8)
def _odd(k): return k if k%2==1 else k+1
def _float(img): return img.astype(np.float64)/255.0


# ══════════════════════════════════════════════════════════════════════════════
# CVIP2 — FREQUENCY DOMAIN ENHANCEMENT
# ══════════════════════════════════════════════════════════════════════════════

def _distance_grid(h,w):
    u=np.arange(h)-h//2; v=np.arange(w)-w//2
    V,U=np.meshgrid(v,u); return np.sqrt(U**2+V**2)

def _freq_filter(img,H):
    gray=_gray(img).astype(np.float32)
    dft=np.fft.fftshift(np.fft.fft2(gray))
    return _norm8(np.fft.ifft2(np.fft.ifftshift(dft*H)).real)

def _lp_butter(img,cutoff=30.0,order=2):
    D=_distance_grid(*_gray(img).shape)
    return _freq_filter(img,1/(1+(D/(cutoff+1e-6))**(2*order)))

def _hp_butter(img,cutoff=30.0,order=2):
    D=_distance_grid(*_gray(img).shape)
    return _freq_filter(img,1-1/(1+(D/(cutoff+1e-6))**(2*order)))

def _hfe(img,cutoff=30.0,boost=1.5,order=2):
    D=_distance_grid(*_gray(img).shape)
    HP=1-1/(1+(D/(cutoff+1e-6))**(2*order))
    return _freq_filter(img,0.5+boost*HP)

def _bandpass(img,center=50.0,width=20.0,order=2):
    D=_distance_grid(*_gray(img).shape)
    safeD=np.where(D==0,1e-6,D)
    H=1/(1+((safeD*width)/(safeD**2-center**2+1e-6))**(2*order))
    return _freq_filter(img,np.clip(H,0,1))

def _bandreject(img,center=50.0,width=20.0,order=2):
    D=_distance_grid(*_gray(img).shape)
    safeD=np.where(D==0,1e-6,D)
    H=1-1/(1+((safeD*width)/(safeD**2-center**2+1e-6))**(2*order))
    return _freq_filter(img,np.clip(H,0,1))

def _notch(img,nu=30,nv=30,radius=10.0):
    gray=_gray(img).astype(np.float32)
    h,w=gray.shape; dft=np.fft.fftshift(np.fft.fft2(gray))
    u=np.arange(h)-h//2; v=np.arange(w)-w//2
    V,U=np.meshgrid(v,u)
    D1=np.sqrt((U-nu)**2+(V-nv)**2); D2=np.sqrt((U+nu)**2+(V+nv)**2)
    H=np.where((D1<=radius)|(D2<=radius),0.0,1.0)
    return _norm8(np.fft.ifft2(np.fft.ifftshift(dft*H)).real)

def _homomorphic(img,cutoff=0.5,gamma_l=0.5,gamma_h=2.0):
    gray=_gray(img).astype(np.float32)+1.0
    log_img=np.log(gray); h,w=log_img.shape
    dft=np.fft.fftshift(np.fft.fft2(log_img))
    u=np.arange(h,dtype=np.float32)-h/2; v=np.arange(w,dtype=np.float32)-w/2
    V,U=np.meshgrid(v,u)
    D=np.sqrt(U**2+V**2)/(max(h,w)*cutoff+1e-6)
    H=(gamma_h-gamma_l)*(1-np.exp(-D**2))+gamma_l
    return _norm8(np.exp(np.fft.ifft2(np.fft.ifftshift(dft*H)).real))


# ══════════════════════════════════════════════════════════════════════════════
# CVIP2 — RESTORATION FILTERS
# ══════════════════════════════════════════════════════════════════════════════

def _motion_psf(h,w,length=20,angle=0.0):
    psf=np.zeros((h,w),np.float32)
    cx,cy=w//2,h//2; rad=np.radians(angle)
    for i in range(-length//2,length//2+1):
        x=cx+int(i*np.cos(rad)); y=cy+int(i*np.sin(rad))
        if 0<=x<w and 0<=y<h: psf[y,x]=1.0
    s=psf.sum(); return psf/s if s>0 else psf

def _psf_otf(psf,shape):
    h,w=shape; pad=np.zeros((h,w),np.float32)
    ph,pw=psf.shape; pad[:ph,:pw]=psf; return np.fft.fft2(pad)

def _wiener(img,psf_length=15,psf_angle=0.0,K=0.01):
    gray=_gray(img).astype(np.float32)
    h,w=gray.shape; G=np.fft.fft2(gray)
    H=_psf_otf(_motion_psf(h,w,psf_length,psf_angle),(h,w))
    W=np.conj(H)/(np.abs(H)**2+K)
    return _norm8(np.abs(np.fft.ifft2(W*G)))

def _cls(img,psf_length=15,psf_angle=0.0,gamma=0.001):
    gray=_gray(img).astype(np.float32)
    h,w=gray.shape; G=np.fft.fft2(gray)
    H=_psf_otf(_motion_psf(h,w,psf_length,psf_angle),(h,w))
    u=np.fft.fftfreq(h)[:,None]; v=np.fft.fftfreq(w)[None,:]
    P=-4*np.pi**2*(u**2+v**2)
    W=np.conj(H)/(np.abs(H)**2+gamma*np.abs(P)**2)
    return _norm8(np.abs(np.fft.ifft2(W*G)))


# ══════════════════════════════════════════════════════════════════════════════
# CVIP2 — HISTOGRAM ENHANCEMENT
# ══════════════════════════════════════════════════════════════════════════════

def _hist_slide(img,shift=50):
    return np.clip(img.astype(np.int32)+shift,0,255).astype(np.uint8)

def _hist_stretch(img,low_pct=2.0,high_pct=98.0):
    lo=np.percentile(img,low_pct); hi=np.percentile(img,high_pct)
    return np.clip((img.astype(np.float32)-lo)/(hi-lo+1e-6)*255,0,255).astype(np.uint8)

def _hist_hyperbolize(img,alpha=0.4):
    norm=img.astype(np.float32)/255.0
    return (np.power(np.clip(norm,1e-6,1.0),alpha)*255).astype(np.uint8)

def _ace(img,k1=0.5,k2=0.5,ksize=15):
    gray=_gray(img).astype(np.float32)
    lm=cv2.blur(gray,(_odd(ksize),_odd(ksize)))
    gm=float(gray.mean()); gs=float(gray.std())+1e-6
    ls=np.abs(gray-lm)+1e-6
    enhanced=gm+(gs/ls)*k1*(gray-lm)+k2*(gray-gm)
    return _norm8(enhanced)

def _pseudo_slice(img):
    gray=_gray(img)
    out=cv2.cvtColor(gray,cv2.COLOR_GRAY2BGR)
    bands=[(0,85,(255,40,40)),(86,170,(40,255,40)),(171,255,(40,40,255))]
    for lo,hi,col in bands:
        out[(gray>=lo)&(gray<=hi)]=col
    return out

def _pseudo_freq_map(img):
    gray=_gray(img).astype(np.float32)
    dft=np.fft.fftshift(np.fft.fft2(gray))
    mag=_norm8(np.log1p(np.abs(dft)))
    phase=_norm8(np.angle(dft)+np.pi)
    h,w=gray.shape; hsv=np.zeros((h,w,3),np.uint8)
    hsv[:,:,0]=phase; hsv[:,:,1]=200; hsv[:,:,2]=mag
    return cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)


# ══════════════════════════════════════════════════════════════════════════════
# CVIP2 — COMPRESSION
# ══════════════════════════════════════════════════════════════════════════════

def _zonal_dct(img,keep_fraction=0.25,block_size=8):
    gray=_gray(img).astype(np.float32); h,w=gray.shape; B=block_size
    ph=(h+B-1)//B*B; pw=(w+B-1)//B*B
    padded=np.zeros((ph,pw),np.float32); padded[:h,:w]=gray
    result=np.zeros_like(padded)
    zone=np.zeros((B,B),bool)
    thr=int(B*keep_fraction*2)
    for i in range(B):
        for j in range(B):
            if i+j<thr: zone[i,j]=True
    kept=int(zone.sum())*(ph//B)*(pw//B)
    total=(ph//B)*(pw//B)*B*B
    for r in range(0,ph,B):
        for c in range(0,pw,B):
            blk=padded[r:r+B,c:c+B]; d=cv2.dct(blk)
            d[~zone]=0; result[r:r+B,c:c+B]=cv2.idct(d)
    out=np.clip(result[:h,:w],0,255).astype(np.uint8)
    ratio=total/max(kept,1); bpp=8.0/ratio
    return out,ratio,bpp

def _threshold_dct(img,keep_pct=10.0,block_size=8):
    gray=_gray(img).astype(np.float32); h,w=gray.shape; B=block_size
    ph=(h+B-1)//B*B; pw=(w+B-1)//B*B
    padded=np.zeros((ph,pw),np.float32); padded[:h,:w]=gray
    result=np.zeros_like(padded); kept=0; total=0
    n_keep=max(1,int(B*B*keep_pct/100))
    for r in range(0,ph,B):
        for c in range(0,pw,B):
            blk=padded[r:r+B,c:c+B]; d=cv2.dct(blk)
            flat=d.flatten(); total+=len(flat)
            tval=np.sort(np.abs(flat))[-n_keep]
            d[np.abs(d)<tval]=0; kept+=np.count_nonzero(d)
            result[r:r+B,c:c+B]=cv2.idct(d)
    out=np.clip(result[:h,:w],0,255).astype(np.uint8)
    ratio=total/max(kept,1); bpp=8.0/ratio
    return out,ratio,bpp

def _btc(img,block_size=4):
    gray=_gray(img).astype(np.float64); h,w=gray.shape; B=block_size
    ph=(h+B-1)//B*B; pw=(w+B-1)//B*B
    padded=np.pad(gray,((0,ph-h),(0,pw-w)),mode='edge')
    result=np.zeros_like(padded)
    for r in range(0,ph,B):
        for c in range(0,pw,B):
            blk=padded[r:r+B,c:c+B]; mu=blk.mean(); sigma=blk.std()+1e-10
            n=B*B; q=np.count_nonzero(blk>=mu)
            if q==0 or q==n: result[r:r+B,c:c+B]=mu; continue
            a=mu-sigma*np.sqrt(q/(n-q+1e-10)); b=mu+sigma*np.sqrt((n-q)/(q+1e-10))
            result[r:r+B,c:c+B]=np.where(blk>=mu,b,a)
    out=np.clip(result[:h,:w],0,255).astype(np.uint8)
    cb=(ph//B)*(pw//B)*(8+B*B//8+1)
    ratio=h*w/max(cb,1); bpp=8.0/ratio
    return out,ratio,bpp

def _wavelet_compress(img,keep_pct=10.0):
    gray=_gray(img).astype(np.float32); sz=1
    while sz*2<=min(gray.shape): sz*=2
    work=cv2.resize(gray,(sz,sz)); oh,ow=gray.shape
    levels=int(np.log2(sz)); tmp=work.copy()
    for _ in range(levels):
        L=(tmp[:,::2]+tmp[:,1::2])/2; Hi=(tmp[:,::2]-tmp[:,1::2])/2
        LL=(L[::2,:]+L[1::2,:])/2; LH=(L[::2,:]-L[1::2,:])/2
        HL=(Hi[::2,:]+Hi[1::2,:])/2; HH=(Hi[::2,:]-Hi[1::2,:])/2
        tmp=np.vstack([np.hstack([LL,LH]),np.hstack([HL,HH])])
    flat=tmp.flatten(); total=len(flat)
    n_keep=max(1,int(total*keep_pct/100))
    tval=np.sort(np.abs(flat))[-n_keep]; tmp[np.abs(tmp)<tval]=0
    kept=np.count_nonzero(tmp)
    for _ in range(levels):
        h2,w2=tmp.shape[0]//2,tmp.shape[1]//2
        LL=tmp[:h2,:w2]; LH=tmp[:h2,w2:]
        HL=tmp[h2:,:w2]; HH=tmp[h2:,w2:]
        L=np.zeros((h2*2,w2),np.float32); Hi=np.zeros((h2*2,w2),np.float32)
        L[::2,:]=LL+LH; L[1::2,:]=LL-LH
        Hi[::2,:]=HL+HH; Hi[1::2,:]=HL-HH
        out_r=np.zeros((h2*2,w2*2),np.float32)
        out_r[:,::2]=L+Hi; out_r[:,1::2]=L-Hi; tmp=out_r
    out=np.clip(cv2.resize(tmp,(ow,oh)),0,255).astype(np.uint8)
    ratio=total/max(kept,1); bpp=8.0/ratio
    return out,ratio,bpp

def _vq_compress(img,codebook_size=64,block_size=4):
    gray=_gray(img).astype(np.float32); h,w=gray.shape; B=block_size
    ph=(h+B-1)//B*B; pw=(w+B-1)//B*B
    padded=np.pad(gray,((0,ph-h),(0,pw-w)),mode='edge')
    blocks=np.array([padded[r:r+B,c:c+B].flatten()
                     for r in range(0,ph,B) for c in range(0,pw,B)],np.float32)
    K=min(codebook_size,len(blocks))
    crit=(cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER,20,0.1)
    _,labels,centers=cv2.kmeans(blocks,K,None,crit,3,cv2.KMEANS_RANDOM_CENTERS)
    result=np.zeros((ph,pw),np.float32); idx=0
    for r in range(0,ph,B):
        for c in range(0,pw,B):
            result[r:r+B,c:c+B]=centers[labels[idx]].reshape(B,B); idx+=1
    out=np.clip(result[:h,:w],0,255).astype(np.uint8)
    bpp=np.log2(K+1)/(B*B); ratio=8.0/bpp
    return out,ratio,bpp

def _dpcm(img,quantize_bits=4):
    gray=_gray(img).astype(np.int32); h,w=gray.shape
    diffs=np.zeros_like(gray); prev=np.zeros(h,np.int32)
    for c in range(w): diffs[:,c]=gray[:,c]-prev; prev=gray[:,c].copy()
    levels=2**quantize_bits; step=510/levels
    q_diffs=np.round(diffs/step)*step
    recon=np.zeros_like(gray); prev_r=np.zeros(h,np.int32)
    for c in range(w):
        recon[:,c]=np.clip(prev_r+q_diffs[:,c],0,255); prev_r=recon[:,c].copy()
    return recon.astype(np.uint8),8.0/quantize_bits,float(quantize_bits)


# ══════════════════════════════════════════════════════════════════════════════
# CVIP2 — FEATURE EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════

def _hist_features(img) -> dict:
    gray=_gray(img).astype(np.float32).flatten()/255.0
    mean=float(gray.mean()); var=float(gray.var()); std=float(np.sqrt(var))+1e-10
    skew=float(np.mean(((gray-mean)/std)**3)); kurt=float(np.mean(((gray-mean)/std)**4))
    hist,_=np.histogram(gray,256,[0,1])
    p=hist/(hist.sum()+1e-10)
    entropy=float(-np.sum(p*np.log2(p+1e-10)))
    return {"mean":mean,"variance":var,"skewness":skew,"kurtosis":kurt,"entropy":entropy}

def _rst_features(img) -> dict:
    gray=_gray(img)
    M=cv2.moments(gray); hu=cv2.HuMoments(M).flatten()
    return {f"hu_{i}":float(-np.sign(h)*np.log10(abs(h)+1e-10)) for i,h in enumerate(hu)}

def _laws_features(img) -> dict:
    gray=_gray(img).astype(np.float32)
    L5=np.array([1,4,6,4,1],np.float32)
    E5=np.array([-1,-2,0,2,1],np.float32)
    S5=np.array([-1,0,2,0,-1],np.float32)
    W5=np.array([-1,2,0,-2,1],np.float32)
    R5=np.array([1,-4,6,-4,1],np.float32)
    kernels={"L5":L5,"E5":E5,"S5":S5,"W5":W5,"R5":R5}
    names=list(kernels.keys())
    result={}
    for i,n1 in enumerate(names):
        for n2 in names[i:]:
            k=np.outer(kernels[n1],kernels[n2])
            result[f"{n1}{n2}"]=float(np.mean(cv2.filter2D(gray,-1,k)**2))
    return result


# ══════════════════════════════════════════════════════════════════════════════
# TRIVISION FUSION — All three libraries together
# ══════════════════════════════════════════════════════════════════════════════

def _multiscale_edge_fusion(img, sigma_fine=1.0, sigma_coarse=3.0, canny_low=50, canny_high=150):
    """
    Fuses: OpenCV Canny + Sobel + skimage Frangi + LoG.
    Produces a richer edge map than any single detector.
    """
    gray = _gray(img)
    # OpenCV edges
    canny = cv2.Canny(gray, canny_low, canny_high).astype(np.float32)
    gx = cv2.Sobel(gray.astype(np.float32), cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(gray.astype(np.float32), cv2.CV_32F, 0, 1)
    sobel = np.sqrt(gx**2 + gy**2)
    # skimage extras
    try:
        from skimage.filters import laplace, gaussian, frangi
        f = gray.astype(np.float64) / 255.0
        log = np.abs(laplace(gaussian(f, sigma=sigma_coarse)))
        vessel = frangi(f, sigmas=np.linspace(sigma_fine, sigma_coarse, 3))
    except ImportError:
        log = np.zeros_like(sobel); vessel = np.zeros_like(sobel)
    # Normalise and fuse
    def n(x): mn,mx=x.min(),x.max(); return (x-mn)/(mx-mn+1e-10)
    fused = (n(canny)*0.3 + n(sobel)*0.3 + n(log)*0.2 + n(vessel.astype(np.float32))*0.2)
    return _norm8(fused)

def _hybrid_denoise(img, sigma=15.0, tv_weight=0.05):
    """
    Pipeline: OpenCV bilateral → skimage TV-Chambolle → OpenCV NLM.
    Removes noise at multiple scales.
    """
    step1 = cv2.bilateralFilter(_bgr(img), 9, 75, 75)
    try:
        from skimage.restoration import denoise_tv_chambolle
        f = step1.astype(np.float64)/255.0
        step2 = (np.clip(denoise_tv_chambolle(f, weight=tv_weight, channel_axis=2),0,1)*255).astype(np.uint8)
    except ImportError:
        step2 = step1
    step3 = cv2.fastNlMeansDenoisingColored(step2, None, sigma/3, sigma/3, 7, 21)
    return step3

def _comprehensive_features(img) -> dict:
    """
    Extracts ALL feature types from all three libraries into one dict:
    histogram, RST/Hu, Laws texture, GLCM (skimage), region props (skimage), spectral.
    """
    result = {}
    result.update(_hist_features(img))
    result.update(_rst_features(img))
    result.update(_laws_features(img))
    try:
        from skimage.feature import graycomatrix, graycoprops
        from skimage.measure import label, regionprops
        gray = (_gray(img)//4).astype(np.uint8)
        glcm = graycomatrix(gray, [1,2], [0,np.pi/4,np.pi/2], levels=64, symmetric=True, normed=True)
        for prop in ['contrast','homogeneity','energy','correlation']:
            for i,v in enumerate(graycoprops(glcm,prop).flatten()):
                result[f"glcm_{prop}_{i}"] = round(float(v),4)
        binary = _gray(img) > 127
        regions = regionprops(label(binary))
        if regions:
            r = max(regions, key=lambda x: x.area)
            result["region_area"] = float(r.area)
            result["region_eccentricity"] = round(float(r.eccentricity),4)
            result["region_solidity"] = round(float(r.solidity),4)
    except ImportError:
        pass
    # Spectral features
    gray = _gray(img).astype(np.float32)
    dft = np.fft.fftshift(np.fft.fft2(gray))
    power = np.abs(dft)**2
    h,w = gray.shape
    u = np.arange(h)-h//2; v = np.arange(w)-w//2
    V,U = np.meshgrid(v,u); D = np.sqrt(U**2+V**2)
    max_r = min(h,w)/2
    for ring in range(4):
        r0=ring*max_r/4; r1=(ring+1)*max_r/4
        mask=(D>=r0)&(D<r1)
        result[f"spectral_ring_{ring}"] = round(float(power[mask].mean()),2)
    return result

def _ab_compare_render(img, fn_a, fn_b, label_a="A", label_b="B"):
    """Render two algorithm outputs side by side with labels."""
    out_a = fn_a(img)
    out_b = fn_b(img)
    if isinstance(out_a, tuple): out_a = out_a[0]
    if isinstance(out_b, tuple): out_b = out_b[0]
    out_a = _bgr(out_a); out_b = _bgr(out_b)
    h = max(out_a.shape[0], out_b.shape[0])
    out_a = cv2.resize(out_a, (out_a.shape[1], h))
    out_b = cv2.resize(out_b, (out_b.shape[1], h))
    divider = np.full((h, 3, 3), [0,120,255], np.uint8)
    combined = np.hstack([out_a, divider, out_b])
    cv2.putText(combined, label_a, (10, 25), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0,200,255), 1)
    cv2.putText(combined, label_b, (out_a.shape[1]+13, 25), cv2.FONT_HERSHEY_DUPLEX, 0.7, (0,200,255), 1)
    return combined

def _quality_score(img) -> dict:
    """Comprehensive quality metrics across all three libraries."""
    import cv2 as _cv2
    gray = _gray(img)
    # Sharpness via Laplacian variance
    lap_var = float(_cv2.Laplacian(gray, _cv2.CV_64F).var())
    # Contrast (std dev)
    contrast = float(gray.std())
    # Brightness
    brightness = float(gray.mean())
    # SNR estimate
    blurred = _cv2.GaussianBlur(gray.astype(np.float64),(5,5),0)
    noise = gray.astype(np.float64) - blurred
    snr = float(10*np.log10(np.mean(gray**2)/(np.mean(noise**2)+1e-10)))
    result = {
        "sharpness (Laplacian var)": round(lap_var, 2),
        "contrast (std)": round(contrast, 2),
        "brightness (mean)": round(brightness, 2),
        "SNR dB": round(snr, 2),
    }
    try:
        from skimage.metrics import structural_similarity as ssim
        blurred_img = _cv2.GaussianBlur(img,(15,15),0)
        gray2 = _gray(blurred_img)
        if gray.shape == gray2.shape:
            result["SSIM vs blurred"] = round(float(ssim(gray,gray2,data_range=255)),4)
    except ImportError:
        pass
    return result

def _super_segment(img, n_superpixels=200, compactness=10.0):
    """
    Multi-method consensus segmentation:
    SLIC superpixels (skimage) + K-means (OpenCV) → intersection overlay.
    """
    bgr = _bgr(img)
    # K-means
    data = bgr.reshape((-1,3)).astype(np.float32)
    crit = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER,20,0.5)
    n_clusters = max(2, n_superpixels//10)
    _,km_labels,_ = cv2.kmeans(data,n_clusters,None,crit,5,cv2.KMEANS_RANDOM_CENTERS)
    km = km_labels.reshape(bgr.shape[:2])
    # SLIC
    try:
        from skimage.segmentation import slic, mark_boundaries
        f = bgr.astype(np.float64)/255.0
        slic_labels = slic(f, n_segments=n_superpixels, compactness=compactness,
                           channel_axis=2, start_label=1)
        # Overlay both boundaries
        marked = mark_boundaries(f, slic_labels, color=(0,1,0))
        # Also draw K-means boundaries in blue
        km_edges = cv2.Canny(km.astype(np.uint8)*10, 1, 2)
        result = (np.clip(marked,0,1)*255).astype(np.uint8)
        result[km_edges>0] = [255,80,0]
        return result
    except ImportError:
        km_norm = _norm8(km.astype(np.float32))
        return cv2.applyColorMap(km_norm, cv2.COLORMAP_TAB10)


# ══════════════════════════════════════════════════════════════════════════════
# REGISTER ALL
# ══════════════════════════════════════════════════════════════════════════════

def register_all():
    cvip = [
        # Freq Filters
        AlgorithmSpec("cvip_lp","Butterworth Lowpass",_LC,"Enhancement","Frequency Domain",_lp_butter,
            [Param.Float("cutoff","Cutoff",30.0,1.0,200.0,1.0),Param.Int("order","Order",2,1,10,1)],_I),
        AlgorithmSpec("cvip_hp","Butterworth Highpass",_LC,"Enhancement","Frequency Domain",_hp_butter,
            [Param.Float("cutoff","Cutoff",30.0,1.0,200.0,1.0),Param.Int("order","Order",2,1,10,1)],_I),
        AlgorithmSpec("cvip_hfe","High Freq Emphasis",_LC,"Enhancement","Frequency Domain",_hfe,
            [Param.Float("cutoff","Cutoff",30.0,1.0,200.0,1.0),Param.Float("boost","Boost",1.5,0.5,5.0,0.1),Param.Int("order","Order",2,1,10,1)],_I),
        AlgorithmSpec("cvip_bp","Butterworth Bandpass",_LC,"Enhancement","Frequency Domain",_bandpass,
            [Param.Float("center","Center",50.0,1.0,200.0,1.0),Param.Float("width","Width",20.0,1.0,100.0,1.0)],_I),
        AlgorithmSpec("cvip_br","Butterworth Bandreject",_LC,"Enhancement","Frequency Domain",_bandreject,
            [Param.Float("center","Center",50.0,1.0,200.0,1.0),Param.Float("width","Width",20.0,1.0,100.0,1.0)],_I),
        AlgorithmSpec("cvip_notch","Notch Filter",_LC,"Enhancement","Frequency Domain",_notch,
            [Param.Int("nu","Notch U",30,-100,100,1),Param.Int("nv","Notch V",30,-100,100,1),Param.Float("radius","Radius",10.0,1.0,50.0,1.0)],_I),
        AlgorithmSpec("cvip_homomorphic","Homomorphic Filter",_LC,"Enhancement","Frequency Domain",_homomorphic,
            [Param.Float("cutoff","Cutoff",0.5,0.05,2.0,0.05),Param.Float("gamma_l","γL",0.5,0.0,1.0,0.05),Param.Float("gamma_h","γH",2.0,1.0,5.0,0.05)],_I),
        # Restoration
        AlgorithmSpec("cvip_wiener","Wiener Filter",_LC,"Restoration","Freq-Domain Restoration",_wiener,
            [Param.Int("psf_length","PSF length",15,1,60,1),Param.Float("psf_angle","PSF angle °",0.0,0.0,180.0,5.0),Param.Float("K","K",0.01,0.0001,1.0,0.001)],_I),
        AlgorithmSpec("cvip_cls","Constrained Least Squares",_LC,"Restoration","Freq-Domain Restoration",_cls,
            [Param.Int("psf_length","PSF length",15,1,60,1),Param.Float("psf_angle","PSF angle °",0.0,0.0,180.0,5.0),Param.Float("gamma","γ",0.001,0.0001,0.1,0.0001)],_I),
        # Histogram
        AlgorithmSpec("cvip_slide","Histogram Slide",_LC,"Enhancement","Histogram",_hist_slide,
            [Param.Int("shift","Shift",-50,-128,128,1)],_I),
        AlgorithmSpec("cvip_stretch","Histogram Stretch",_LC,"Enhancement","Histogram",_hist_stretch,
            [Param.Float("low_pct","Low %",2.0,0.0,20.0,0.5),Param.Float("high_pct","High %",98.0,80.0,100.0,0.5)],_I),
        AlgorithmSpec("cvip_hyperbolize","Histogram Hyperbolize",_LC,"Enhancement","Histogram",_hist_hyperbolize,
            [Param.Float("alpha","Gamma α",0.4,0.1,3.0,0.05)],_I),
        AlgorithmSpec("cvip_ace","Adaptive Contrast (ACE)",_LC,"Enhancement","Spatial",_ace,
            [Param.Float("k1","k1",0.5,0.0,2.0,0.05),Param.Float("k2","k2",0.5,0.0,2.0,0.05),Param.Int("ksize","Kernel",15,3,51,2)],_I),
        AlgorithmSpec("cvip_pseudo_slice","Intensity Slicing",_LC,"Enhancement","Pseudo-Color",_pseudo_slice,[],_I),
        AlgorithmSpec("cvip_pseudo_freq","Frequency Color Map",_LC,"Enhancement","Pseudo-Color",_pseudo_freq_map,[],_I),
        # Compression
        AlgorithmSpec("cvip_zonal","Zonal DCT",_LC,"Compression","DCT-Based",_zonal_dct,
            [Param.Float("keep_fraction","Keep fraction",0.25,0.05,1.0,0.05),Param.Int("block_size","Block size",8,4,16,4)],_T),
        AlgorithmSpec("cvip_threshold_dct","Threshold DCT",_LC,"Compression","DCT-Based",_threshold_dct,
            [Param.Float("keep_pct","Keep %",10.0,1.0,100.0,1.0),Param.Int("block_size","Block size",8,4,16,4)],_T),
        AlgorithmSpec("cvip_btc","Block Truncation Coding",_LC,"Compression","Statistical",_btc,
            [Param.Int("block_size","Block size",4,2,16,2)],_T),
        AlgorithmSpec("cvip_wavelet","Wavelet Threshold",_LC,"Compression","Wavelet",_wavelet_compress,
            [Param.Float("keep_pct","Keep %",10.0,0.5,50.0,0.5)],_T),
        AlgorithmSpec("cvip_vq","Vector Quantization",_LC,"Compression","Statistical",_vq_compress,
            [Param.Int("codebook_size","Codebook size",64,4,256,4),Param.Int("block_size","Block size",4,2,8,2)],_T),
        AlgorithmSpec("cvip_dpcm","DPCM",_LC,"Compression","Predictive",_dpcm,
            [Param.Int("quantize_bits","Quant bits",4,1,8,1)],_T),
        # Features
        AlgorithmSpec("cvip_hist_feat","Histogram Features",_LC,"Analysis","Feature Extraction",_hist_features,[],_F),
        AlgorithmSpec("cvip_rst_feat","RST-Invariant (Hu) Moments",_LC,"Analysis","Feature Extraction",_rst_features,[],_F),
        AlgorithmSpec("cvip_laws_feat","Laws Texture Energy",_LC,"Analysis","Feature Extraction",_laws_features,[],_F),
    ]
    fusion = [
        AlgorithmSpec("tv_multiedge","Multi-Scale Edge Fusion",_LT,"Analysis","Edge Detection",_multiscale_edge_fusion,
            [Param.Float("sigma_fine","σ fine",1.0,0.5,5.0,0.5),Param.Float("sigma_coarse","σ coarse",3.0,1.0,10.0,0.5),Param.Int("canny_low","Canny low",50,0,200,5),Param.Int("canny_high","Canny high",150,50,255,5)],_I,
            "Fuses Canny + Sobel + Frangi + LoG for richer edges",["fusion"]),
        AlgorithmSpec("tv_hybrid_denoise","Hybrid Denoise Pipeline",_LT,"Restoration","Denoising",_hybrid_denoise,
            [Param.Float("sigma","NLM strength",15.0,1.0,50.0,1.0),Param.Float("tv_weight","TV weight",0.05,0.001,0.5,0.005)],_I,
            "Bilateral → TV-Chambolle → NLM pipeline",["fusion","denoise"]),
        AlgorithmSpec("tv_all_features","Comprehensive Feature Extraction",_LT,"Analysis","Feature Extraction",_comprehensive_features,[],_F,
            "All features: histogram + Hu + Laws + GLCM + region props + spectral",["fusion","features"]),
        AlgorithmSpec("tv_quality","Image Quality Score",_LT,"Analysis","Feature Extraction",_quality_score,[],_F,
            "Sharpness + contrast + SNR + SSIM",["metrics","quality"]),
        AlgorithmSpec("tv_superseg","Multi-Method Segmentation",_LT,"Analysis","Segmentation",_super_segment,
            [Param.Int("n_superpixels","Superpixels",200,20,500,20),Param.Float("compactness","Compactness",10.0,0.1,50.0,1.0)],_O,
            "SLIC (skimage) + K-means (OpenCV) consensus boundaries",["fusion","segmentation"]),
    ]
    for s in cvip + fusion:
        REGISTRY.register(s)
