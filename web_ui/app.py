"""
TriVision Web UI — Flask Application
Runs in any browser. No PyQt6 / display needed.
Works on Raspberry Pi, servers, Docker, or any machine.

Start:
    python web_ui/app.py
    # or
    python web_ui/app.py --host 0.0.0.0 --port 5000

Then open: http://localhost:5000
"""

import sys
import os
import base64
import json
import time
import threading
import argparse

# ── Path bootstrap ────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_socketio import SocketIO, emit

# ── TriVision core ────────────────────────────────────────────────────────────
from core.registry import REGISTRY, Lib, ReturnType
import core.algorithms_opencv as _ocv
import core.algorithms_skimage as _ski
import core.algorithms_cvip_fusion as _cvf
from pipeline.engine import Pipeline
from plugins.sdk import load_plugins

# Bootstrap
_ocv.register_all()
_ski.register_all()
_cvf.register_all()
load_plugins(os.path.join(os.path.dirname(os.path.dirname(__file__)), "plugins"))

# ── App setup ─────────────────────────────────────────────────────────────────
app = Flask(__name__, template_folder="templates", static_folder="static")
# SECURITY FIX: Load SECRET_KEY from environment or generate secure random key
app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY") or os.urandom(32).hex()
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024  # 32 MB upload limit
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="eventlet")

# ── Global state ──────────────────────────────────────────────────────────────
# BUG FIX: Added TTL-based session management to prevent memory leak
_sessions: dict[str, dict] = {}          # session_id → {input_img, output_img, pipeline, last_access}
_sessions_lock = threading.Lock()
SESSION_TTL = 1800  # 30 minutes in seconds
MAX_SESSIONS = 100  # Maximum number of concurrent sessions

_webcam_thread = None
_webcam_lock   = threading.Lock()
_webcam_running = False
_webcam_algo_key = None
_webcam_algo_params = {}

UPLOAD_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "uploads")
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "outputs")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _img_to_b64(img: np.ndarray, quality: int = 85) -> str:
    """Encode numpy BGR image to base64 JPEG string for browser display."""
    if img is None:
        return ""
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode()

def _b64_to_img(b64: str) -> np.ndarray:
    """Decode base64 image string from browser to numpy BGR."""
    if "," in b64:
        b64 = b64.split(",")[1]
    buf = base64.b64decode(b64)
    arr = np.frombuffer(buf, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)

def _cleanup_sessions():
    """Remove expired sessions to prevent memory leak."""
    with _sessions_lock:
        now = time.time()
        expired = [sid for sid, sess in _sessions.items()
                   if now - sess.get("last_access", 0) > SESSION_TTL]
        for sid in expired:
            del _sessions[sid]

        # If still over limit, remove oldest sessions
        if len(_sessions) > MAX_SESSIONS:
            sorted_sessions = sorted(_sessions.items(),
                                    key=lambda x: x[1].get("last_access", 0))
            for sid, _ in sorted_sessions[:len(_sessions) - MAX_SESSIONS]:
                del _sessions[sid]

def _get_session(sid: str) -> dict:
    _cleanup_sessions()  # Clean up on every access
    with _sessions_lock:
        if sid not in _sessions:
            _sessions[sid] = {"input_img": None, "output_img": None,
                               "pipeline": Pipeline(), "last_access": time.time()}
        else:
            _sessions[sid]["last_access"] = time.time()
    return _sessions[sid]

def _run_algo(img: np.ndarray, key: str, params: dict):
    """Run one algorithm and return result image + metadata."""
    spec = REGISTRY.get(key)
    if spec is None:
        return img, "Unknown algorithm"
    t0 = time.perf_counter()
    result = spec.fn(img, **params)
    elapsed = (time.perf_counter() - t0) * 1000
    message = f"{spec.label}  •  {elapsed:.0f}ms"
    if isinstance(result, tuple) and len(result) == 3:
        out_img, ratio, bpp = result
        message += f"  •  ratio {ratio:.2f}:1  •  {bpp:.2f} bpp"
        return out_img, message
    if isinstance(result, dict):
        return None, result      # feature dict
    if isinstance(result, np.ndarray) and len(result.shape) == 2:
        result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
    return result, message

def _compute_metrics(orig: np.ndarray, proc: np.ndarray) -> dict:
    if orig is None or proc is None:
        return {}
    def gray(x): return cv2.cvtColor(x, cv2.COLOR_BGR2GRAY) if len(x.shape)==3 else x
    g1 = gray(orig).astype(np.float64)
    g2 = gray(cv2.resize(proc, (orig.shape[1], orig.shape[0]))).astype(np.float64)
    mse  = float(np.mean((g1-g2)**2))
    psnr = float(10*np.log10(255**2/(mse+1e-10)))
    rmse = float(np.sqrt(mse))
    sharp = float(cv2.Laplacian(proc, cv2.CV_64F).var())
    result = {"psnr": round(psnr,2), "rmse": round(rmse,2), "sharpness": round(sharp,2)}
    try:
        from skimage.metrics import structural_similarity as ssim
        result["ssim"] = round(float(ssim(g1, g2, data_range=255.0)), 4)
    except Exception:
        pass
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# REST API Routes
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/health")
def health():
    return jsonify({"status": "ok", "algorithms": len(REGISTRY)})

@app.route("/api/algorithms")
def get_algorithms():
    """Return full algorithm tree for the sidebar."""
    tree = {}
    for spec in sorted(REGISTRY.all(), key=lambda s: (s.category, s.subcategory, s.label)):
        cat  = spec.category
        sub  = spec.subcategory
        tree.setdefault(cat, {}).setdefault(sub, []).append({
            "key":         spec.key,
            "label":       spec.label,
            "lib":         spec.lib.value,
            "description": spec.description,
            "return_type": spec.return_type.value,
            "params": [
                {
                    "name":    p.name,
                    "label":   p.label,
                    "kind":    p.kind,
                    "default": p.default,
                    "lo":      p.lo,
                    "hi":      p.hi,
                    "step":    p.step,
                    "choices": p.choices,
                }
                for p in spec.params
            ],
        })
    return jsonify(tree)

@app.route("/api/upload", methods=["POST"])
def upload_image():
    """Accept image upload from browser."""
    sid = request.form.get("session_id", "default")
    sess = _get_session(sid)

    if "file" in request.files:
        f = request.files["file"]
        buf = np.frombuffer(f.read(), np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    elif "b64" in request.form:
        img = _b64_to_img(request.form["b64"])
    else:
        return jsonify({"error": "No image provided"}), 400

    if img is None:
        return jsonify({"error": "Cannot decode image"}), 400

    sess["input_img"] = img
    h, w = img.shape[:2]
    return jsonify({
        "success": True,
        "preview": _img_to_b64(img),
        "width": w, "height": h,
        "channels": img.shape[2] if len(img.shape)==3 else 1,
    })

@app.route("/api/process", methods=["POST"])
def process_image():
    """Apply one algorithm to the current input image."""
    data = request.get_json()
    sid    = data.get("session_id", "default")
    key    = data.get("algo_key")
    params = data.get("params", {})

    sess = _get_session(sid)
    img  = sess.get("input_img")
    if img is None:
        return jsonify({"error": "No input image. Upload one first."}), 400

    result, meta = _run_algo(img, key, params)

    # Feature dict case
    if result is None and isinstance(meta, dict):
        return jsonify({"success": True, "type": "features", "features": meta})

    if result is None:
        return jsonify({"error": str(meta)}), 500

    sess["output_img"] = result
    metrics = _compute_metrics(img, result)
    h, w = result.shape[:2]

    return jsonify({
        "success":  True,
        "type":     "image",
        "preview":  _img_to_b64(result),
        "message":  meta,
        "metrics":  metrics,
        "width":    w,
        "height":   h,
    })

@app.route("/api/pipeline/run", methods=["POST"])
def run_pipeline():
    """Run the session pipeline end-to-end."""
    data = request.get_json()
    sid  = data.get("session_id", "default")
    sess = _get_session(sid)
    img  = sess.get("input_img")
    if img is None:
        return jsonify({"error": "No input image"}), 400

    pipeline = sess["pipeline"]
    if not pipeline.nodes:
        return jsonify({"error": "Pipeline is empty. Add algorithms first."}), 400

    t0 = time.perf_counter()
    result = pipeline.final_output(img)
    elapsed = (time.perf_counter() - t0) * 1000

    if isinstance(result, dict):
        return jsonify({"success": True, "type": "features", "features": result})

    if not isinstance(result, np.ndarray):
        return jsonify({"error": "Pipeline returned unexpected type"}), 500

    if len(result.shape) == 2:
        result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)

    sess["output_img"] = result
    metrics = _compute_metrics(img, result)

    return jsonify({
        "success": True,
        "type": "image",
        "preview": _img_to_b64(result),
        "message": f"Pipeline '{pipeline.name}'  •  {len(pipeline.nodes)} steps  •  {elapsed:.0f}ms",
        "metrics": metrics,
        "steps": len(pipeline.nodes),
    })

@app.route("/api/pipeline/add", methods=["POST"])
def pipeline_add():
    data = request.get_json()
    sid  = data.get("session_id", "default")
    key  = data.get("algo_key")
    sess = _get_session(sid)
    p    = sess["pipeline"]
    upstream = p.nodes[-1].node_id if p.nodes else None
    node = p.add_node(key, upstream_id=upstream)
    spec = REGISTRY.get(key)
    return jsonify({"success": True, "node_id": node.node_id,
                    "label": spec.label if spec else key})

@app.route("/api/pipeline/remove", methods=["POST"])
def pipeline_remove():
    data = request.get_json()
    sid     = data.get("session_id", "default")
    node_id = data.get("node_id")
    sess = _get_session(sid)
    sess["pipeline"].remove_node(node_id)
    return jsonify({"success": True})

@app.route("/api/pipeline/clear", methods=["POST"])
def pipeline_clear():
    sid = request.get_json().get("session_id", "default")
    _get_session(sid)["pipeline"].clear()
    return jsonify({"success": True})

@app.route("/api/pipeline/list")
def pipeline_list():
    sid  = request.args.get("session_id", "default")
    sess = _get_session(sid)
    return jsonify({"nodes": [
        {"node_id": n.node_id, "label": REGISTRY.get(n.algo_key).label
         if REGISTRY.get(n.algo_key) else n.algo_key}
        for n in sess["pipeline"].nodes
    ]})

@app.route("/api/output/download")
def download_output():
    """Download the current output image as PNG."""
    from flask import Response
    sid  = request.args.get("session_id", "default")
    sess = _get_session(sid)
    img  = sess.get("output_img")
    if img is None:
        return jsonify({"error": "No output image"}), 404
    _, buf = cv2.imencode(".png", img)
    return Response(buf.tobytes(), mimetype="image/png",
                    headers={"Content-Disposition": "attachment; filename=trivision_output.png"})

@app.route("/api/use_output_as_input", methods=["POST"])
def use_output_as_input():
    sid  = request.get_json().get("session_id", "default")
    sess = _get_session(sid)
    out  = sess.get("output_img")
    if out is None:
        return jsonify({"error": "No output image"}), 400
    sess["input_img"] = out.copy()
    h, w = out.shape[:2]
    return jsonify({"success": True, "preview": _img_to_b64(out), "width": w, "height": h})

@app.route("/api/diff", methods=["POST"])
def diff_images():
    sid  = request.get_json().get("session_id", "default")
    sess = _get_session(sid)
    inp  = sess.get("input_img")
    out  = sess.get("output_img")
    if inp is None or out is None:
        return jsonify({"error": "Need both input and output"}), 400
    r = cv2.resize(out, (inp.shape[1], inp.shape[0]))
    diff = cv2.convertScaleAbs(cv2.absdiff(inp, r), alpha=3.0)
    sess["output_img"] = diff
    return jsonify({"success": True, "preview": _img_to_b64(diff)})

@app.route("/api/preset/<name>", methods=["POST"])
def load_preset(name):
    presets = {
        "edge":        Pipeline.edge_detection_pipeline,
        "denoise":     Pipeline.denoise_and_enhance,
        "segment":     Pipeline.segmentation_pipeline,
        "features":    Pipeline.feature_extraction_pipeline,
        "compression": Pipeline.compression_benchmark,
    }
    if name not in presets:
        return jsonify({"error": f"Unknown preset '{name}'"}), 404
    sid  = request.get_json().get("session_id", "default")
    sess = _get_session(sid)
    p    = presets[name]()
    sess["pipeline"] = p
    spec_map = {n.node_id: (REGISTRY.get(n.algo_key).label
                             if REGISTRY.get(n.algo_key) else n.algo_key)
                for n in p.nodes}
    return jsonify({"success": True, "name": p.name,
                    "nodes": [{"node_id": k, "label": v} for k, v in spec_map.items()]})

@app.route("/api/features/extract", methods=["POST"])
def extract_features():
    sid  = request.get_json().get("session_id", "default")
    sess = _get_session(sid)
    img  = sess.get("input_img")
    if img is None:
        return jsonify({"error": "No input image"}), 400
    spec = REGISTRY.get("tv_all_features")
    if spec is None:
        return jsonify({"error": "Feature extractor not available"}), 500
    features = spec.fn(img)
    return jsonify({"success": True, "features": features})

@app.route("/api/default_image", methods=["POST"])
def default_image():
    """Generate and return the default TriVision demo image."""
    img = np.zeros((420, 640, 3), np.uint8)
    for i in range(0, 640, 40): cv2.line(img, (i,0), (i,420), (8,10,18), 1)
    for j in range(0, 420, 40): cv2.line(img, (0,j), (640,j), (8,10,18), 1)
    cv2.rectangle(img,(30,30),(220,200),(0,180,120),2)
    cv2.circle(img,(460,130),85,(180,80,220),-1)
    cv2.ellipse(img,(320,310),(135,65),20,0,360,(0,160,255),3)
    cv2.line(img,(30,370),(610,380),(220,200,0),3)
    cv2.putText(img,"TriVision",(155,265),cv2.FONT_HERSHEY_DUPLEX,1.6,(240,240,255),2)
    cv2.putText(img,"Web UI",(230,300),cv2.FONT_HERSHEY_SIMPLEX,0.7,(100,130,180),1)

    sid  = request.get_json().get("session_id", "default")
    sess = _get_session(sid)
    sess["input_img"] = img
    h, w = img.shape[:2]
    return jsonify({"success": True, "preview": _img_to_b64(img), "width": w, "height": h})


# ═══════════════════════════════════════════════════════════════════════════════
# WebSocket — Webcam Streaming
# ═══════════════════════════════════════════════════════════════════════════════

@socketio.on("webcam_start")
def handle_webcam_start(data):
    global _webcam_running, _webcam_thread, _webcam_algo_key, _webcam_algo_params
    # BUG FIX: Added missing global declarations - these were local variables before
    cam_idx = int(data.get("camera_index", 0))
    _webcam_algo_key    = data.get("algo_key", None)
    _webcam_algo_params = data.get("params", {})

    with _webcam_lock:
        if _webcam_running:
            return
        _webcam_running = True

    def stream():
        global _webcam_running
        cap = cv2.VideoCapture(cam_idx)
        if not cap.isOpened():
            socketio.emit("webcam_error", {"message": f"Cannot open camera {cam_idx}"})
            _webcam_running = False
            return
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        frame_count = 0
        t_start = time.time()
        while _webcam_running:
            ret, frame = cap.read()
            if not ret: break
            # Apply live algorithm if set
            ak = _webcam_algo_key
            if ak:
                spec = REGISTRY.get(ak)
                if spec and spec.return_type in (ReturnType.IMAGE, ReturnType.OVERLAY):
                    try:
                        r = spec.fn(frame, **_webcam_algo_params)
                        if isinstance(r, tuple): r = r[0]
                        if isinstance(r, np.ndarray):
                            frame = cv2.cvtColor(r, cv2.COLOR_GRAY2BGR) if len(r.shape)==2 else r
                    except Exception:
                        pass
            frame_count += 1
            fps = frame_count / max(time.time()-t_start, 0.001)
            b64 = _img_to_b64(frame, quality=70)
            socketio.emit("webcam_frame", {"frame": b64, "fps": round(fps, 1)})
            socketio.sleep(0.033)  # ~30 fps
        cap.release()
        _webcam_running = False
        socketio.emit("webcam_stopped", {})

    _webcam_thread = socketio.start_background_task(stream)

@socketio.on("webcam_stop")
def handle_webcam_stop(data=None):
    global _webcam_running
    _webcam_running = False

@socketio.on("webcam_set_algo")
def handle_webcam_algo(data):
    global _webcam_algo_key, _webcam_algo_params
    _webcam_algo_key    = data.get("algo_key")
    _webcam_algo_params = data.get("params", {})

@socketio.on("webcam_snapshot")
def handle_webcam_snapshot(data):
    """Capture current webcam frame and make it the session input."""
    sid = data.get("session_id", "default")
    b64 = data.get("frame")
    if b64:
        img = _b64_to_img(b64)
        if img is not None:
            _get_session(sid)["input_img"] = img
            socketio.emit("snapshot_ready", {
                "preview": _img_to_b64(img),
                "width": img.shape[1],
                "height": img.shape[0],
            })


# ═══════════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TriVision Web UI")
    parser.add_argument("--host",  default="127.0.0.1", help="Host to bind (0.0.0.0 for LAN)")
    parser.add_argument("--port",  default=5000, type=int, help="Port number")
    parser.add_argument("--debug", action="store_true",   help="Enable Flask debug mode")
    args = parser.parse_args()

    print(f"\n{'='*55}")
    print(f"  TriVision Web UI")
    print(f"  {len(REGISTRY)} algorithms loaded")
    print(f"  Open in browser: http://{args.host}:{args.port}")
    print(f"  LAN access:      http://<your-ip>:{args.port}")
    print(f"{'='*55}\n")

    socketio.run(app, host=args.host, port=args.port, debug=args.debug)
