"""
TriVision — Visual Pipeline Engine

Algorithms can be chained as a DAG of PipelineNode objects.
Each node caches its output and only recomputes when inputs or params change.
The engine supports:
  - Linear chains
  - Branching (one image → two paths → A/B comparison)
  - Feature accumulation nodes (collect dicts from multiple upstream nodes)
  - Serialise / deserialise pipeline to/from JSON
"""

from __future__ import annotations
import json
import hashlib
import numpy as np
import cv2
from typing import Any, Optional
from dataclasses import dataclass, field
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.registry import REGISTRY, AlgorithmSpec, ReturnType


# ─── Pipeline Node ────────────────────────────────────────────────────────────

@dataclass
class PipelineNode:
    """One algorithm step in the pipeline."""
    node_id: str           # unique id, e.g. "node_0"
    algo_key: str          # key into REGISTRY
    params: dict           # {param_name: value}
    upstream_ids: list[str] = field(default_factory=list)  # list of upstream node_ids

    # Runtime cache (not serialised)
    _result: Any = field(default=None, repr=False, compare=False)
    _param_hash: str = field(default="", repr=False, compare=False)

    @property
    def spec(self) -> Optional[AlgorithmSpec]:
        return REGISTRY.get(self.algo_key)

    @property
    def label(self) -> str:
        return self.spec.label if self.spec else self.algo_key

    def _hash_params(self) -> str:
        return hashlib.md5(json.dumps(self.params, sort_keys=True).encode()).hexdigest()[:8]

    def execute(self, input_img: np.ndarray) -> Any:
        """Execute the algorithm, using cache if params haven't changed."""
        ph = self._hash_params()
        if self._result is not None and self._param_hash == ph:
            return self._result  # cache hit
        spec = self.spec
        if spec is None:
            self._result = input_img; return input_img
        try:
            result = spec.fn(input_img, **self.params)
            # Normalise compression tuple — extract image
            if isinstance(result, tuple) and len(result) == 3:
                img, ratio, bpp = result
                # Store metadata on the array for later reporting
                if isinstance(img, np.ndarray):
                    img = img.copy()
                result = img
            self._result = result
            self._param_hash = ph
        except Exception as e:
            # Return an error image rather than crashing
            h, w = input_img.shape[:2]
            err = input_img.copy()
            cv2.putText(err, f"ERR: {str(e)[:60]}", (5, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
            self._result = err
        return self._result

    def invalidate(self):
        self._result = None
        self._param_hash = ""

    def to_dict(self) -> dict:
        return {
            "node_id": self.node_id,
            "algo_key": self.algo_key,
            "params": self.params,
            "upstream_ids": self.upstream_ids,
        }

    @staticmethod
    def from_dict(d: dict) -> "PipelineNode":
        spec = REGISTRY.get(d["algo_key"])
        params = d.get("params", {})
        if spec:
            defaults = {p.name: p.default for p in spec.params}
            defaults.update(params)
            params = defaults
        return PipelineNode(d["node_id"], d["algo_key"], params, d.get("upstream_ids", []))


# ─── Pipeline ─────────────────────────────────────────────────────────────────

class Pipeline:
    """
    DAG of PipelineNodes. Linear pipelines are the common case;
    branching is supported for A/B comparisons.
    """

    def __init__(self, name: str = "Untitled Pipeline"):
        self.name = name
        self._nodes: dict[str, PipelineNode] = {}
        self._order: list[str] = []   # topological order
        self._counter = 0

    # ── Node management ───────────────────────────────────────────────

    def add_node(self, algo_key: str, params: dict = None,
                  upstream_id: str = None) -> PipelineNode:
        """Add a new algorithm node, optionally connected to an upstream node."""
        nid = f"node_{self._counter}"; self._counter += 1
        spec = REGISTRY.get(algo_key)
        if spec is None:
            raise ValueError(f"Unknown algorithm: {algo_key}")
        defaults = {p.name: p.default for p in spec.params}
        if params:
            defaults.update(params)
        upstream = [upstream_id] if upstream_id else []
        node = PipelineNode(nid, algo_key, defaults, upstream)
        self._nodes[nid] = node
        self._order.append(nid)
        return node

    def remove_node(self, node_id: str):
        if node_id in self._nodes:
            del self._nodes[node_id]
            self._order = [n for n in self._order if n != node_id]
            # Remove this node from upstreams of other nodes
            for n in self._nodes.values():
                n.upstream_ids = [u for u in n.upstream_ids if u != node_id]

    def update_params(self, node_id: str, params: dict):
        if node_id in self._nodes:
            self._nodes[node_id].params.update(params)
            self._nodes[node_id].invalidate()

    def get_node(self, node_id: str) -> Optional[PipelineNode]:
        return self._nodes.get(node_id)

    @property
    def nodes(self) -> list[PipelineNode]:
        return [self._nodes[nid] for nid in self._order if nid in self._nodes]

    def clear(self):
        self._nodes.clear(); self._order.clear(); self._counter = 0

    # ── Execution ─────────────────────────────────────────────────────

    def run(self, input_img: np.ndarray) -> dict[str, Any]:
        """
        Execute the pipeline in topological order.
        Returns {node_id: result} for all nodes.
        """
        results: dict[str, Any] = {}
        results["__input__"] = input_img

        for nid in self._order:
            if nid not in self._nodes:
                continue
            node = self._nodes[nid]
            # Determine input: take last upstream result, else pipeline input
            if node.upstream_ids:
                upstream = node.upstream_ids[-1]
                img = results.get(upstream, input_img)
            else:
                img = input_img
            # If upstream returned a dict (features), pass original input
            if isinstance(img, dict):
                img = input_img
            results[nid] = node.execute(img)

        return results

    def run_to(self, node_id: str, input_img: np.ndarray) -> Any:
        """Run pipeline up to and including the given node."""
        results = self.run(input_img)
        return results.get(node_id)

    def final_output(self, input_img: np.ndarray) -> Any:
        """Return the result of the last node."""
        if not self._order:
            return input_img
        results = self.run(input_img)
        for nid in reversed(self._order):
            if nid in results:
                return results[nid]
        return input_img

    # ── A/B comparison ────────────────────────────────────────────────

    def ab_compare(self, input_img: np.ndarray,
                    node_a: str, node_b: str) -> np.ndarray:
        """Return a side-by-side comparison image of two node outputs."""
        results = self.run(input_img)
        out_a = results.get(node_a, input_img)
        out_b = results.get(node_b, input_img)

        def to_bgr(x):
            if isinstance(x, dict):
                # Feature dict — render as text image
                h, w = 200, 350
                img = np.zeros((h, w, 3), np.uint8)
                for i, (k, v) in enumerate(list(x.items())[:12]):
                    cv2.putText(img, f"{k}: {v:.4f}" if isinstance(v, float) else f"{k}: {v}",
                                (5, 18 + i*15), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (180,220,180), 1)
                return img
            if len(x.shape) == 2:
                return cv2.cvtColor(x, cv2.COLOR_GRAY2BGR)
            return x

        a = to_bgr(out_a); b = to_bgr(out_b)
        h = max(a.shape[0], b.shape[0])
        a = cv2.resize(a, (a.shape[1], h)); b = cv2.resize(b, (b.shape[1], h))
        divider = np.full((h, 4, 3), [40, 160, 255], np.uint8)
        combined = np.hstack([a, divider, b])
        spec_a = REGISTRY.get(self._nodes[node_a].algo_key)
        spec_b = REGISTRY.get(self._nodes[node_b].algo_key)
        la = spec_a.label if spec_a else node_a
        lb = spec_b.label if spec_b else node_b
        cv2.putText(combined, la, (6, 22), cv2.FONT_HERSHEY_DUPLEX, 0.6, (40,200,255), 1)
        cv2.putText(combined, lb, (a.shape[1]+10, 22), cv2.FONT_HERSHEY_DUPLEX, 0.6, (40,200,255), 1)
        return combined

    # ── Serialisation ─────────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "counter": self._counter,
            "order": self._order,
            "nodes": {nid: node.to_dict() for nid, node in self._nodes.items()},
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @staticmethod
    def from_dict(d: dict) -> "Pipeline":
        p = Pipeline(d.get("name", "Untitled"))
        p._counter = d.get("counter", 0)
        p._order = d.get("order", [])
        for nid, nd in d.get("nodes", {}).items():
            p._nodes[nid] = PipelineNode.from_dict(nd)
        return p

    @staticmethod
    def from_json(s: str) -> "Pipeline":
        return Pipeline.from_dict(json.loads(s))

    def save(self, path: str):
        with open(path, "w") as f:
            f.write(self.to_json())

    @staticmethod
    def load(path: str) -> "Pipeline":
        with open(path) as f:
            return Pipeline.from_json(f.read())

    # ── Presets ───────────────────────────────────────────────────────

    @staticmethod
    def edge_detection_pipeline() -> "Pipeline":
        p = Pipeline("Edge Detection Pipeline")
        n1 = p.add_node("ocv_clahe", {"clip": 2.0})
        n2 = p.add_node("tv_multiedge", {"sigma_fine": 1.0, "sigma_coarse": 3.0},
                         upstream_id=n1.node_id)
        return p

    @staticmethod
    def denoise_and_enhance() -> "Pipeline":
        p = Pipeline("Denoise + Enhance Pipeline")
        n1 = p.add_node("tv_hybrid_denoise", {"sigma": 15.0})
        n2 = p.add_node("cvip_ace", {"k1": 0.5, "k2": 0.5}, upstream_id=n1.node_id)
        n3 = p.add_node("ocv_unsharp", {"sigma": 1.0, "strength": 1.2}, upstream_id=n2.node_id)
        return p

    @staticmethod
    def segmentation_pipeline() -> "Pipeline":
        p = Pipeline("Segmentation Pipeline")
        n1 = p.add_node("ocv_bilateral", {"d": 9, "sigma_color": 75.0, "sigma_space": 75.0})
        n2 = p.add_node("tv_superseg", {"n_superpixels": 200}, upstream_id=n1.node_id)
        return p

    @staticmethod
    def compression_benchmark() -> "Pipeline":
        p = Pipeline("Compression Benchmark")
        p.add_node("cvip_zonal", {"keep_fraction": 0.25})
        p.add_node("cvip_btc", {"block_size": 4})
        p.add_node("ocv_jpeg", {"quality": 20})
        p.add_node("cvip_wavelet", {"keep_pct": 10.0})
        return p

    @staticmethod
    def feature_extraction_pipeline() -> "Pipeline":
        p = Pipeline("Feature Extraction Pipeline")
        n1 = p.add_node("ocv_clahe", {"clip": 2.0})
        p.add_node("tv_all_features", {}, upstream_id=n1.node_id)
        return p
