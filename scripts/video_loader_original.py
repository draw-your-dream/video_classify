"""Video frame loader.

Decodes a mp4 with PyAV (cpu fallback) and returns numpy arrays of
sampled frames. Two sampling strategies:

- ``dense``: every N-th frame, used for optical flow and SSIM where
  consecutive-frame deltas matter. Default N=4 → ~30 frames from a 121-
  frame clip.
- ``sparse``: K evenly-spaced frames covering the full clip, used for
  CLIP/pose/VLM where temporal range matters more than density. Default
  K=8.

Frames are returned as ``np.uint8`` HWC RGB. Optional resize keeps the
aspect ratio by short-side scaling.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import av
import numpy as np


@dataclass
class LoadedVideo:
    frames_dense: np.ndarray  # (Td, H, W, 3) uint8 RGB
    frames_sparse: np.ndarray  # (Ts, H, W, 3) uint8 RGB
    fps: float
    n_frames_total: int
    dense_indices: list[int]
    sparse_indices: list[int]


def _short_side_resize(img: np.ndarray, short_side: int | None) -> np.ndarray:
    if short_side is None:
        return img
    h, w = img.shape[:2]
    s = short_side / min(h, w)
    if abs(s - 1.0) < 1e-3:
        return img
    new_w = int(round(w * s))
    new_h = int(round(h * s))
    # av.VideoFrame.reformat would be faster but we already have ndarray.
    import cv2

    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


def load_video(
    path: str | Path,
    dense_stride: int = 4,
    sparse_count: int = 999,  # use all frames by default
    short_side: int | None = 256,
    sparse_short_side: int | None = 224,
) -> LoadedVideo:
    """Decode a video and return dense + sparse frame batches.

    Single decode pass: walk every frame once, collect frames whose
    index falls into either the dense or sparse selection.
    """
    path = str(path)

    # First pass: get frame count and fps without full decode where possible.
    with av.open(path) as container:
        stream = container.streams.video[0]
        # PyAV may give 0 here for variable-rate streams. Fallback below.
        n_frames = int(stream.frames or 0)
        fps_val = stream.average_rate
        fps = float(fps_val) if fps_val else 30.0

    # Sparse indices are evenly spaced; dense are every-Nth.
    if n_frames <= 0:
        # We'll discover n_frames on the fly.
        sparse_indices: list[int] = []
        dense_indices: list[int] = []
    else:
        if sparse_count >= n_frames:
            sparse_indices = list(range(n_frames))
        else:
            sparse_indices = np.linspace(
                0, n_frames - 1, sparse_count, dtype=int
            ).tolist()
        dense_indices = list(range(0, n_frames, dense_stride))

    # Second pass: decode and collect.
    dense: list[np.ndarray] = []
    sparse: list[np.ndarray] = []
    collected_idx_dense: list[int] = []
    collected_idx_sparse: list[int] = []

    sparse_set = set(sparse_indices)
    dense_set = set(dense_indices)

    with av.open(path) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        idx = 0
        for frame in container.decode(stream):
            want_dense = (idx in dense_set) if dense_set else (
                idx % dense_stride == 0
            )
            want_sparse = idx in sparse_set
            if want_dense or want_sparse:
                img = frame.to_ndarray(format="rgb24")
                if want_dense:
                    dense.append(_short_side_resize(img, short_side))
                    collected_idx_dense.append(idx)
                if want_sparse:
                    sparse.append(_short_side_resize(img, sparse_short_side))
                    collected_idx_sparse.append(idx)
            idx += 1

    # If sparse_set was empty (unknown frame count), pick after the fact.
    if not sparse and dense:
        # rebuild sparse from collected dense by even spacing
        if len(dense) <= sparse_count:
            sparse = list(dense)
            collected_idx_sparse = list(collected_idx_dense)
        else:
            picks = np.linspace(0, len(dense) - 1, sparse_count, dtype=int).tolist()
            sparse = [dense[p] for p in picks]
            collected_idx_sparse = [collected_idx_dense[p] for p in picks]
        # If sparse short_side differs, re-resize from dense (rough).
        if sparse_short_side is not None and sparse_short_side != short_side:
            sparse = [_short_side_resize(f, sparse_short_side) for f in sparse]

    if not dense:
        raise RuntimeError(f"no frames decoded from {path}")

    frames_dense = np.stack(dense, axis=0)
    frames_sparse = np.stack(sparse, axis=0) if sparse else frames_dense[:1]
    n_total = idx

    return LoadedVideo(
        frames_dense=frames_dense,
        frames_sparse=frames_sparse,
        fps=fps,
        n_frames_total=n_total,
        dense_indices=collected_idx_dense,
        sparse_indices=collected_idx_sparse,
    )
