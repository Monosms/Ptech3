import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import cv2

# Lazy imports for heavy deps
try:
    import torch
except Exception:
    torch = None  # type: ignore


SAM_URLS = {
    "vit_b": "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
}


def ensure_dir(path: str) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def download_file(url: str, dest_path: str) -> None:
    import urllib.request

    ensure_dir(str(Path(dest_path).parent))
    # Stream download to avoid memory spikes
    with urllib.request.urlopen(url) as response, open(dest_path, "wb") as out_file:
        chunk_size = 1 << 20  # 1MB
        while True:
            chunk = response.read(chunk_size)
            if not chunk:
                break
            out_file.write(chunk)


def download_sam_checkpoint(checkpoint_path: str, variant: str = "vit_b") -> None:
    url = SAM_URLS.get(variant)
    if url is None:
        raise ValueError(f"Unsupported SAM variant: {variant}")
    print(f"[INFO] Downloading SAM {variant} checkpoint to {checkpoint_path} ...")
    download_file(url, checkpoint_path)
    print("[INFO] Download complete.")


def load_sam_automatic_mask_generator(checkpoint_path: str, device: Optional[str] = None, variant: str = "vit_b"):
    """Load SAM and return an AutomaticMaskGenerator instance."""
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

    if device is None:
        if torch is not None and torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

    model_type = variant
    sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
    sam.to(device)

    # Conservative defaults to balance quality/speed on CPU
    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=16,
        pred_iou_thresh=0.86,
        stability_score_thresh=0.92,
        crop_n_layers=1,
        crop_n_points_downscale_factor=2,
        min_mask_region_area=300,  # filter small regions
    )
    return mask_generator, device


def detect_face_roi(bgr_image: np.ndarray) -> Tuple[Optional[Tuple[int, int, int, int]], Tuple[int, int, int, int]]:
    """Detect face and compute a top-of-head ROI above the forehead.

    Returns (face_rect or None, top_roi_rect)
    Rect format: (x, y, w, h)
    """
    gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    # Haar cascade path bundled with OpenCV
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    face_cascade = cv2.CascadeClassifier(cascade_path)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, flags=cv2.CASCADE_SCALE_IMAGE, minSize=(60, 60))

    h, w = gray.shape[:2]

    if len(faces) == 0:
        # Fallback: assume face roughly at vertical center, top ROI at upper third
        face_rect = None
        roi_w = int(0.5 * w)
        roi_h = int(0.18 * h)
        roi_x = max(0, (w - roi_w) // 2)
        roi_y = max(0, int(0.12 * h))
        top_roi = (roi_x, roi_y, roi_w, roi_h)
        return face_rect, top_roi

    # Choose the largest face
    areas = [fw * fh for (_, _, fw, fh) in faces]
    idx = int(np.argmax(areas))
    fx, fy, fw, fh = faces[idx]

    # Define a ROI above the forehead
    roi_w = int(fw * 1.2)
    roi_h = int(fh * 0.45)
    roi_x = max(0, fx + fw // 2 - roi_w // 2)
    roi_y = max(0, fy - int(roi_h * 0.9))

    # Clamp to image bounds
    roi_x = int(np.clip(roi_x, 0, w - 1))
    roi_y = int(np.clip(roi_y, 0, h - 1))
    roi_w = int(np.clip(roi_w, 1, w - roi_x))
    roi_h = int(np.clip(roi_h, 1, h - roi_y))

    return (fx, fy, fw, fh), (roi_x, roi_y, roi_w, roi_h)


def compute_mask_score(mask: np.ndarray, top_roi: Tuple[int, int, int, int], face_rect: Optional[Tuple[int, int, int, int]], image_shape: Tuple[int, int, int]) -> float:
    h, w = image_shape[:2]
    x, y, rw, rh = top_roi

    mask_bool = mask.astype(bool)

    # Overlap with top ROI
    top_mask = np.zeros((h, w), dtype=bool)
    top_mask[y : y + rh, x : x + rw] = True

    top_overlap = (mask_bool & top_mask).sum()
    top_fraction = top_overlap / (rw * rh + 1e-6)

    # Penalize overlap with face area (hair shouldn't cover most of face)
    face_penalty = 0.0
    if face_rect is not None:
        fx, fy, fw, fh = face_rect
        face_mask = np.zeros((h, w), dtype=bool)
        face_mask[fy : fy + fh, fx : fx + fw] = True
        face_overlap = (mask_bool & face_mask).sum()
        face_fraction = face_overlap / (fw * fh + 1e-6)
        # Moderate penalty
        face_penalty = 0.6 * face_fraction

    # Area prior: hair area typically moderate fraction of image
    area_fraction = mask_bool.sum() / (h * w + 1e-6)
    area_score = 1.0 - abs(area_fraction - 0.16) / 0.16  # peak at ~16%
    area_score = float(np.clip(area_score, 0.0, 1.0))

    # Spatial proximity to face center (if face known)
    spatial_bonus = 0.0
    if face_rect is not None:
        fx, fy, fw, fh = face_rect
        face_cx = fx + fw / 2.0
        mask_coords = np.column_stack(np.nonzero(mask_bool))
        if mask_coords.size > 0:
            mask_cy, mask_cx = mask_coords.mean(axis=0)
            dx = abs(mask_cx - face_cx) / (w + 1e-6)
            spatial_bonus = float(1.0 - np.clip(dx / 0.25, 0.0, 1.0)) * 0.3

    score = (1.2 * top_fraction) + (0.6 * area_score) + spatial_bonus - face_penalty
    return float(score)


def select_hair_mask(masks: list, image_rgb: np.ndarray, top_roi, face_rect) -> Optional[np.ndarray]:
    if not masks:
        return None
    h, w = image_rgb.shape[:2]

    best_score = -1e9
    best_mask = None

    for m in masks:
        # Each m: dict with keys 'segmentation', 'area', 'stability_score', 'predicted_iou', etc.
        seg = m.get("segmentation")
        if seg is None:
            continue
        score = compute_mask_score(seg.astype(np.uint8), top_roi, face_rect, (h, w, 3))
        # Slightly favor higher confidence/stability
        score += 0.2 * float(m.get("predicted_iou", 0.0))
        score += 0.2 * float(m.get("stability_score", 0.0))

        if score > best_score:
            best_score = score
            best_mask = seg.astype(np.uint8)

    return best_mask


def compute_metrics(hair_mask: np.ndarray, face_rect: Optional[Tuple[int, int, int, int]], top_roi: Tuple[int, int, int, int]) -> Dict[str, float]:
    h, w = hair_mask.shape[:2]
    mask = (hair_mask > 0).astype(np.uint8)

    # Left/Right split around face center or image center
    if face_rect is not None:
        fx, fy, fw, fh = face_rect
        cx = fx + fw // 2
    else:
        cx = w // 2

    left_mask = mask[:, :cx]
    right_mask = mask[:, cx:]

    left_area = float(left_mask.sum())
    right_area = float(right_mask.sum())

    # Normalize by side area to get density
    left_density = left_area / (h * max(cx, 1))
    right_density = right_area / (h * max(w - cx, 1))
    asymmetry_index = abs(left_density - right_density) / (max((left_density + right_density) / 2.0, 1e-6))

    # Frontal coverage in top ROI and corner deficit
    tx, ty, tw, th = top_roi
    top_mask = mask[ty : ty + th, tx : tx + tw]
    coverage_top = float(top_mask.sum()) / (tw * th + 1e-6)

    # Split top ROI into left corner, center, right corner
    third = max(tw // 3, 1)
    left_top = top_mask[:, :third]
    center_top = top_mask[:, third : 2 * third]
    right_top = top_mask[:, 2 * third :]

    left_cov = float(left_top.sum()) / (left_top.size + 1e-6)
    center_cov = float(center_top.sum()) / (center_top.size + 1e-6)
    right_cov = float(right_top.sum()) / (right_top.size + 1e-6)

    corner_cov = (left_cov + right_cov) / 2.0
    corner_deficit = max(0.0, center_cov - corner_cov)

    # Normalize risk components (heuristic)
    risk_from_low_coverage = max(0.0, (0.5 - coverage_top) / 0.5)  # 0 if >=0.5, 1 if 0
    risk_from_asymmetry = min(1.0, asymmetry_index / 0.3)  # 0.3 ~ noticeable
    risk_from_corners = min(1.0, corner_deficit / 0.3)

    risk = 0.5 * risk_from_low_coverage + 0.3 * risk_from_corners + 0.2 * risk_from_asymmetry
    risk = float(np.clip(risk, 0.0, 1.0))

    return {
        "coverage_top": float(coverage_top),
        "left_density": float(left_density),
        "right_density": float(right_density),
        "asymmetry_index": float(asymmetry_index),
        "corner_deficit": float(corner_deficit),
        "risk_score": float(risk),  # 0.0 (낮음) ~ 1.0 (높음)
    }


def visualize_and_save_outputs(out_dir: str, bgr: np.ndarray, hair_mask: Optional[np.ndarray], metrics: Dict[str, float]) -> None:
    ensure_dir(out_dir)
    base = Path(out_dir)

    cv2.imwrite(str(base / "input.jpg"), bgr)

    if hair_mask is None:
        with open(base / "metrics.json", "w", encoding="utf-8") as f:
            json.dump({"error": "no_hair_mask", "metrics": metrics}, f, ensure_ascii=False, indent=2)
        return

    mask_u8 = (hair_mask > 0).astype(np.uint8) * 255
    cv2.imwrite(str(base / "hair_mask.png"), mask_u8)

    # Cutout (BGR)
    cutout = cv2.bitwise_and(bgr, bgr, mask=mask_u8)
    cv2.imwrite(str(base / "hair_cutout.png"), cutout)

    # RGBA cutout (transparent background)
    b, g, r = cv2.split(bgr)
    rgba = cv2.merge((b, g, r, mask_u8))
    cv2.imwrite(str(base / "hair_cutout_rgba.png"), rgba)

    # Overlay boundary
    contours, _ = cv2.findContours((mask_u8 > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    overlay = bgr.copy()
    cv2.drawContours(overlay, contours, -1, (0, 255, 0), thickness=2)
    alpha = 0.35
    blended = cv2.addWeighted(overlay, alpha, bgr, 1 - alpha, 0)
    cv2.imwrite(str(base / "overlay.png"), blended)

    with open(base / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)


def extract_hairline_polyline(hair_mask: np.ndarray, top_roi: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
    """Estimate hairline as the lowest hair boundary within the top ROI.

    Returns an Nx2 int array of (x, y) points or None if not found.
    """
    x, y, w, h = top_roi
    if w <= 2 or h <= 2:
        return None

    roi = (hair_mask[y : y + h, x : x + w] > 0)
    hR, wR = roi.shape[:2]
    xs: list[int] = []
    ys: list[float] = []

    for col in range(wR):
        col_data = roi[:, col]
        idxs = np.flatnonzero(col_data[::-1])
        if idxs.size == 0:
            ys.append(np.nan)
            xs.append(col)
        else:
            y_rel_from_bottom = int(idxs[0])
            y_rel = (hR - 1) - y_rel_from_bottom
            ys.append(float(y_rel))
            xs.append(col)

    xs_arr = np.asarray(xs)
    ys_arr = np.asarray(ys, dtype=float)
    valid = ~np.isnan(ys_arr)

    # Find contiguous valid segments and pick the main central one
    segments: list[Tuple[int, int]] = []
    start: Optional[int] = None
    for i, v in enumerate(valid):
        if v and start is None:
            start = i
        elif (not v) and (start is not None):
            segments.append((start, i - 1))
            start = None
    if start is not None:
        segments.append((start, wR - 1))

    if not segments:
        return None

    mid = wR // 2
    def seg_key(seg: Tuple[int, int]) -> Tuple[int, int, int]:
        s, e = seg
        crosses_mid = 1 if (s <= mid <= e) else 0
        length = e - s + 1
        center_dist = -abs(((s + e) // 2) - mid)
        return (crosses_mid, length, center_dist)

    s, e = max(segments, key=seg_key)
    xs_seg = xs_arr[s : e + 1]
    ys_seg = ys_arr[s : e + 1]

    # Simple moving-average smoothing with edge padding
    k =  nine_if_possible = min(11, max(3, (len(xs_seg) // 10) * 2 + 1))
    k = max(3, nine_if_possible if nine_if_possible % 2 == 1 else nine_if_possible - 1)
    if len(xs_seg) >= k:
        pad = k // 2
        ys_pad = np.pad(ys_seg, (pad, pad), mode="edge")
        kernel = np.ones(k, dtype=float) / float(k)
        ys_smooth = np.convolve(ys_pad, kernel, mode="valid")
    else:
        ys_smooth = ys_seg

    pts = np.column_stack([xs_seg + x, ys_smooth + y]).astype(np.int32)
    if pts.shape[0] < 2:
        return None
    return pts


def save_hairline_outputs(base_dir: str, bgr: np.ndarray, hair_mask: np.ndarray, top_roi: Tuple[int, int, int, int], hairline_pts: Optional[np.ndarray]) -> None:
    base = Path(base_dir)
    if hairline_pts is None:
        return

    # Save polyline points
    pts_list = [{"x": int(p[0]), "y": int(p[1])} for p in hairline_pts.tolist()]
    with open(base / "hairline_points.json", "w", encoding="utf-8") as f:
        json.dump({"top_roi": {"x": top_roi[0], "y": top_roi[1], "w": top_roi[2], "h": top_roi[3]}, "points": pts_list}, f, ensure_ascii=False, indent=2)

    # Mask (thin line) and band (thicker for visualization)
    h, w = hair_mask.shape[:2]
    thin = np.zeros((h, w), dtype=np.uint8)
    band = np.zeros((h, w), dtype=np.uint8)
    cv2.polylines(thin, [hairline_pts], isClosed=False, color=255, thickness=2, lineType=cv2.LINE_AA)
    cv2.polylines(band, [hairline_pts], isClosed=False, color=255, thickness=6, lineType=cv2.LINE_AA)

    # Keep band inside hair to avoid skin bleed
    band = cv2.bitwise_and(band, (hair_mask > 0).astype(np.uint8) * 255)

    cv2.imwrite(str(base / "hairline_mask.png"), thin)
    cv2.imwrite(str(base / "hairline_band.png"), band)

    # RGBA band cutout (transparent background outside band)
    b_ch, g_ch, r_ch = cv2.split(bgr)
    band_rgba = cv2.merge((b_ch, g_ch, r_ch, band))
    cv2.imwrite(str(base / "hairline_band_rgba.png"), band_rgba)

    # Cutout of hairline band
    hairline_cutout = cv2.bitwise_and(bgr, bgr, mask=band)
    cv2.imwrite(str(base / "hairline_cutout.png"), hairline_cutout)

    # Overlay line in red on original image
    overlay = bgr.copy()
    cv2.polylines(overlay, [hairline_pts], isClosed=False, color=(0, 0, 255), thickness=2, lineType=cv2.LINE_AA)
    blended = cv2.addWeighted(overlay, 0.5, bgr, 0.5, 0)
    cv2.imwrite(str(base / "hairline_overlay.png"), blended)


def run(image_path: str, output_dir: str, sam_checkpoint: str, variant: str = "vit_b", device: Optional[str] = None, force_download: bool = False) -> None:
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Read image (BGR)
    bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"Failed to read image: {image_path}")

    # Face ROI
    face_rect, top_roi = detect_face_roi(bgr)

    # Ensure checkpoint
    if force_download or (not os.path.isfile(sam_checkpoint)):
        ensure_dir(str(Path(sam_checkpoint).parent))
        download_sam_checkpoint(sam_checkpoint, variant)

    # Load SAM automatic mask generator
    mask_generator, device_used = load_sam_automatic_mask_generator(sam_checkpoint, device=device, variant=variant)

    # SAM expects RGB
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    masks = mask_generator.generate(rgb)

    hair_mask = select_hair_mask(masks, rgb, top_roi, face_rect)

    # Hairline extraction within the top-of-head ROI
    hairline_pts = None
    if hair_mask is not None:
        try:
            hairline_pts = extract_hairline_polyline(hair_mask.astype(np.uint8), top_roi)
        except Exception:
            hairline_pts = None

    metrics = compute_metrics(hair_mask if hair_mask is not None else np.zeros(rgb.shape[:2], dtype=np.uint8), face_rect, top_roi)

    visualize_and_save_outputs(output_dir, bgr, hair_mask, metrics)
    if hair_mask is not None:
        save_hairline_outputs(output_dir, bgr, hair_mask.astype(np.uint8), top_roi, hairline_pts)

    print("[OK] Outputs saved to:", output_dir)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Initial hair loss PoC using SAM (ViT-B) + OpenCV")
    p.add_argument("--image", required=True, help="Path to input image")
    p.add_argument("--output-dir", default="outputs", help="Directory to save outputs")
    p.add_argument("--sam-checkpoint", default=str(Path("weights") / "sam_vit_b_01ec64.pth"), help="Path to SAM ViT-B checkpoint .pth")
    p.add_argument("--variant", default="vit_b", choices=["vit_b"], help="SAM model variant")
    p.add_argument("--device", default=None, choices=["cpu", "cuda", None], help="Device to run SAM on")
    p.add_argument("--download", action="store_true", help="Force download SAM checkpoint")
    return p.parse_args(argv)


if __name__ == "__main__":
    args = parse_args()
    try:
        run(
            image_path=args.image,
            output_dir=args.output_dir,
            sam_checkpoint=args.sam_checkpoint,
            variant=args.variant,
            device=args.device,
            force_download=args.download,
        )
    except Exception as e:
        print(f"[ERROR] {e}")
        sys.exit(1)
