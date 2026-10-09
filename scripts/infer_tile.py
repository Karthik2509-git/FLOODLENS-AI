"""
FLOODLENS-AI — Milestone 1: Sentinel-1 Flood Segmentation Inference & Validation Pipeline
Performs genuine model inference using pretrained Radiance ViT-MAE (SEN1Floods11 v2).
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
from typing import Dict, Any, Tuple

import numpy as np
import rasterio
import torch
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from PIL import Image

# Add models/radiance to Python search path
BASE_DIR = Path(__file__).resolve().parent.parent
RADIANCE_SRC = BASE_DIR / "models" / "radiance"
DATA_SAMPLE_DIR = BASE_DIR / "data" / "sample"
if str(RADIANCE_SRC) not in sys.path:
    sys.path.insert(0, str(RADIANCE_SRC))

try:
    from model import RadianceMAE, RadianceSegModel
except ImportError as e:
    raise ImportError(f"Failed to import Radiance model architecture from {RADIANCE_SRC}: {e}")

# SSL4EO-S12 normalisation constants documented in model card
SSL4EO_S12_MEAN = np.array([-12.577, -20.265], dtype=np.float32)  # [VV, VH]
SSL4EO_S12_STD  = np.array([5.176, 5.870], dtype=np.float32)     # [VV, VH]
DB_CLIP_MIN = -60.0
DB_CLIP_MAX = 20.0
CROP_SIZE = 224
DEFAULT_STRIDE = 112


def load_model(checkpoint_path: Path, device: torch.device) -> RadianceSegModel:
    """Load Radiance ViT-B encoder and segmentation head from checkpoint."""
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")
    
    print(f"[*] Initializing RadianceMAE ViT-B (patch=16, in_chans=2)...")
    encoder = RadianceMAE(img_size=CROP_SIZE, patch_size=16, in_chans=2)
    model = RadianceSegModel(encoder)
    
    print(f"[*] Loading checkpoint: {checkpoint_path.name} ({checkpoint_path.stat().st_size / 1e6:.1f} MB)...")
    ckpt = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    
    state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    missing, unexpected = model.load_state_dict(state_dict, strict=True)
    print(f"[OK] Model state loaded strictly without mismatch (missing={len(missing)}, unexpected={len(unexpected)}).")
    
    model.to(device)
    model.eval()
    return model


def preprocess_sar(raw_sar: np.ndarray) -> np.ndarray:
    """
    Applies exact documented preprocessing:
    1. nan_to_num (SAR nodata handling)
    2. Clip to [-60.0, +20.0] dB
    3. Z-score normalization using SSL4EO-S12 stats
    """
    # Shape: (2, H, W)
    x = np.nan_to_num(raw_sar.astype(np.float32), nan=0.0)
    x = np.clip(x, DB_CLIP_MIN, DB_CLIP_MAX)
    
    # Standardize per channel
    x[0] = (x[0] - SSL4EO_S12_MEAN[0]) / SSL4EO_S12_STD[0]
    x[1] = (x[1] - SSL4EO_S12_MEAN[1]) / SSL4EO_S12_STD[1]
    return x


def infer_center_crop(model: torch.nn.Module, sar_norm: np.ndarray, device: torch.device) -> np.ndarray:
    """Infers the center 224x224 crop."""
    _, H, W = sar_norm.shape
    start_y = (H - CROP_SIZE) // 2
    start_x = (W - CROP_SIZE) // 2
    
    crop = sar_norm[:, start_y:start_y + CROP_SIZE, start_x:start_x + CROP_SIZE]
    tensor = torch.from_numpy(crop).unsqueeze(0).to(device)  # (1, 2, 224, 224)
    
    with torch.no_grad():
        logits = model(tensor)
        prob = torch.sigmoid(logits).squeeze().cpu().numpy()
        
    return prob


def infer_full_tile_sliding_window(
    model: torch.nn.Module,
    sar_norm: np.ndarray,
    device: torch.device,
    stride: int = DEFAULT_STRIDE
) -> np.ndarray:
    """
    Infers the full tile using overlapping 224x224 sliding windows and averages probabilities.
    Guarantees full coverage up to the tile edges.
    """
    _, H, W = sar_norm.shape
    prob_accum = np.zeros((H, W), dtype=np.float32)
    count_accum = np.zeros((H, W), dtype=np.float32)
    
    # Calculate coordinate offsets
    def get_offsets(dimension: int, window: int, step: int):
        offsets = list(range(0, dimension - window + 1, step))
        if offsets[-1] != dimension - window:
            offsets.append(dimension - window)
        return offsets

    y_offsets = get_offsets(H, CROP_SIZE, stride)
    x_offsets = get_offsets(W, CROP_SIZE, stride)
    
    patches = []
    coords = []
    
    for y in y_offsets:
        for x in x_offsets:
            patch = sar_norm[:, y:y + CROP_SIZE, x:x + CROP_SIZE]
            patches.append(patch)
            coords.append((y, x))
            
    # Batch inference
    batch_tensor = torch.from_numpy(np.stack(patches)).to(device)  # (N, 2, 224, 224)
    
    with torch.no_grad():
        logits = model(batch_tensor)
        probs = torch.sigmoid(logits).squeeze(1).cpu().numpy()  # (N, 224, 224)
        
    for i, (y, x) in enumerate(coords):
        prob_accum[y:y + CROP_SIZE, x:x + CROP_SIZE] += probs[i]
        count_accum[y:y + CROP_SIZE, x:x + CROP_SIZE] += 1.0
        
    full_prob = prob_accum / np.maximum(count_accum, 1.0)
    return full_prob


def compute_metrics(pred_binary: np.ndarray, gt: np.ndarray) -> Dict[str, Any]:
    """
    Computes validation metrics against ground truth label:
    gt == 1: Flooded
    gt == 0: Non-flooded land
    gt == -1: NoData / unannotated (ignored)
    """
    valid_mask = (gt == 0) | (gt == 1)
    n_valid = int(np.sum(valid_mask))
    
    if n_valid == 0:
        return {"error": "No valid labeled pixels found in ground truth."}
    
    p = pred_binary[valid_mask].astype(bool)
    g = (gt[valid_mask] == 1).astype(bool)
    
    tp = int(np.sum(p & g))
    fp = int(np.sum(p & ~g))
    fn = int(np.sum(~p & g))
    tn = int(np.sum(~p & ~g))
    
    iou = tp / (tp + fp + fn + 1e-8)
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
    accuracy = (tp + tn) / (tp + fp + fn + tn + 1e-8)
    
    return {
        "valid_pixels": n_valid,
        "ground_truth_flood_pixels": int(np.sum(g)),
        "predicted_flood_pixels": int(np.sum(p)),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "true_negatives": tn,
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "accuracy": float(accuracy)
    }


def save_visualizations(
    output_dir: Path,
    raw_sar: np.ndarray,
    full_prob: np.ndarray,
    binary_mask: np.ndarray,
    threshold: float,
    gt: np.ndarray = None
):
    """Generates clean output visualizations: flood mask and comparison overlay."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Save standalone flood mask (PNG)
    mask_colored = np.zeros((*binary_mask.shape, 4), dtype=np.uint8)
    # Bright cyan for flood: RGBA (0, 220, 255, 255)
    mask_colored[binary_mask == 1] = [0, 225, 255, 255]
    mask_colored[binary_mask == 0] = [15, 23, 42, 255]  # Deep dark slate for dry
    
    mask_img = Image.fromarray(mask_colored)
    mask_path = output_dir / "flood_mask.png"
    mask_img.save(mask_path)
    print(f"[OK] Saved flood mask: {mask_path}")
    
    # 2. Save comprehensive multi-panel overlay figure
    # Prepare VV SAR grayscale visualization
    vv_raw = raw_sar[0].copy()
    vv_valid = np.nan_to_num(vv_raw, nan=-30.0)
    vv_min, vv_max = -25.0, -5.0
    vv_norm = np.clip((vv_valid - vv_min) / (vv_max - vv_min), 0.0, 1.0)
    
    fig, axes = plt.subplots(1, 4 if gt is not None else 3, figsize=(20, 5), dpi=150)
    
    # Panel 1: SAR VV Backscatter
    axes[0].imshow(vv_norm, cmap="gray")
    axes[0].set_title("Sentinel-1 SAR (VV dB)", fontsize=12, fontweight="bold")
    axes[0].axis("off")
    
    # Panel 2: Predicted Flood Probability Heatmap
    im_prob = axes[1].imshow(full_prob, cmap="magma", vmin=0.0, vmax=1.0)
    axes[1].set_title("Predicted Water Probability", fontsize=12, fontweight="bold")
    axes[1].axis("off")
    fig.colorbar(im_prob, ax=axes[1], fraction=0.046, pad=0.04, label="Probability")
    
    # Panel 3: SAR with Predicted Flood Overlay
    axes[2].imshow(vv_norm, cmap="gray")
    # Overlay flood mask in transparent cyan
    overlay_rgba = np.zeros((*binary_mask.shape, 4), dtype=np.float32)
    overlay_rgba[binary_mask == 1] = [0.0, 0.88, 1.0, 0.65]  # Cyan with 65% alpha
    axes[2].imshow(overlay_rgba)
    axes[2].set_title(f"Flood Overlay (Threshold={threshold:.2f})", fontsize=12, fontweight="bold")
    axes[2].axis("off")
    
    # Panel 4: Comparison with Ground Truth (if available)
    if gt is not None:
        # Error map: TP (green), FP (red), FN (yellow), TN (gray)
        valid_mask = (gt == 0) | (gt == 1)
        comp_img = np.zeros((*binary_mask.shape, 3), dtype=np.uint8)
        comp_img[~valid_mask] = [40, 40, 40]                  # NoData (dark gray)
        comp_img[valid_mask & (gt == 0) & (binary_mask == 0)] = [100, 100, 100] # TN (gray)
        comp_img[valid_mask & (gt == 1) & (binary_mask == 1)] = [34, 197, 94]   # TP (Green)
        comp_img[valid_mask & (gt == 0) & (binary_mask == 1)] = [239, 68, 68]   # FP (Red)
        comp_img[valid_mask & (gt == 1) & (binary_mask == 0)] = [234, 179, 8]   # FN (Yellow)
        
        axes[3].imshow(comp_img)
        axes[3].set_title("Validation: TP(Green) FP(Red) FN(Yellow)", fontsize=12, fontweight="bold")
        axes[3].axis("off")
        
    plt.tight_layout()
    overlay_path = output_dir / "flood_overlay.png"
    plt.savefig(overlay_path, bbox_inches="tight")
    plt.close()
    print(f"[OK] Saved composite overlay visualization: {overlay_path}")


def get_working_device(requested_device: str = "auto") -> Tuple[torch.device, str]:
    """Determines working device and safely probes CUDA compatibility (e.g. for sm_120 Blackwell GPUs)."""
    if requested_device == "cpu":
        return torch.device("cpu"), "Host CPU"
        
    if requested_device in ["auto", "cuda"] and torch.cuda.is_available():
        try:
            # Probe CUDA execution with a small convolution/tensor op
            test_x = torch.zeros(1, 2, 16, 16, device="cuda")
            test_conv = torch.nn.Conv2d(2, 2, 3, padding=1).cuda()
            _ = test_conv(test_x)
            device_name = torch.cuda.get_device_name(0)
            return torch.device("cuda"), f"GPU: {device_name}"
        except Exception as e:
            print(f"[!] Warning: CUDA device detected ({torch.cuda.get_device_name(0)}) but failed kernel probe ({e}).")
            print(f"[*] Gracefully falling back to high-performance Host CPU execution.")
            return torch.device("cpu"), "Host CPU (CUDA sm kernel mismatch fallback)"
            
    return torch.device("cpu"), "Host CPU"


def compute_geodesic_mask_area(
    mask: np.ndarray,
    transform,
    crs: rasterio.crs.CRS,
    valid_mask: np.ndarray = None
) -> Dict[str, Any]:
    """
    Computes rigorous geodesic surface area for a binary mask in km^2 and m^2.
    - For Geographic CRS (e.g., EPSG:4326): Uses WGS84 ellipsoidal row-wise integration
      accounting for meridional and parallel radii of curvature (latitude convergence).
    - For Projected CRS (e.g., UTM in meters): Uses affine pixel dimensions |dx * dy|.
    - Explicitly excludes NoData / invalid pixels when valid_mask is provided.
    """
    if valid_mask is not None:
        effective_mask = (mask > 0) & valid_mask
    else:
        effective_mask = (mask > 0)
        
    H, W = mask.shape
    dx = abs(transform[0])
    dy = abs(transform[4])
    
    if crs.is_geographic:
        # WGS84 Ellipsoidal parameters
        a = 6378137.0          # Semi-major axis in meters
        e2 = 0.00669437999014  # First eccentricity squared
        
        top_y = transform[5]
        dy_signed = transform[4]  # Typically negative
        row_indices = np.arange(H, dtype=np.float64)
        row_lats = top_y + (row_indices + 0.5) * dy_signed
        row_lats_rad = np.radians(row_lats)
        
        sin_lat = np.sin(row_lats_rad)
        cos_lat = np.cos(row_lats_rad)
        denom = 1.0 - e2 * (sin_lat ** 2)
        
        # Radii of curvature for 1 degree
        m_per_deg_lat = (np.pi / 180.0) * a * (1.0 - e2) / (denom ** 1.5)
        m_per_deg_lon = (np.pi / 180.0) * a * cos_lat / np.sqrt(denom)
        
        row_pixel_area_m2 = (dx * m_per_deg_lon) * (dy * m_per_deg_lat)
        row_counts = np.sum(effective_mask, axis=1)
        total_area_m2 = float(np.sum(row_counts * row_pixel_area_m2))
        total_area_km2 = float(total_area_m2 / 1.0e6)
        mean_pixel_area_m2 = float(np.mean(row_pixel_area_m2))
        method_desc = "WGS84 ellipsoidal geodesic row-integration (latitudinal convergence)"
    else:
        pixel_area_m2 = float(dx * dy)
        total_pixels = int(np.sum(effective_mask))
        total_area_m2 = float(total_pixels * pixel_area_m2)
        total_area_km2 = float(total_area_m2 / 1.0e6)
        mean_pixel_area_m2 = pixel_area_m2
        method_desc = "Projected planar raster area (|dx * dy| in CRS units)"
        
    return {
        "estimated_flood_area_km2": round(total_area_km2, 6),
        "estimated_flood_area_m2": round(total_area_m2, 2),
        "mean_pixel_area_m2": round(mean_pixel_area_m2, 4),
        "active_flood_pixels": int(np.sum(effective_mask)),
        "calculation_method": method_desc,
        "is_estimate": True
    }


def run_inference_pipeline(
    sar_path: Path,
    label_path: Path = None,
    checkpoint_path: Path = RADIANCE_SRC / "best.pt",
    output_dir: Path = BASE_DIR / "outputs",
    threshold: float = 0.25,
    device_name: str = "auto"
):
    start_time = time.time()
    
    # Robust Device selection
    device, device_desc = get_working_device(device_name)
    print(f"[*] Execution device: {device} ({device_desc})")
    
    # Load Model
    model = load_model(checkpoint_path, device)
    
    # Read SAR GeoTIFF
    print(f"[*] Reading SAR raster: {sar_path}...")
    with rasterio.open(sar_path) as src:
        raw_sar = src.read()
        crs_obj = src.crs
        crs_str = str(src.crs)
        transform_tuple = tuple(src.transform)
        bounds_dict = {
            "left": src.bounds.left,
            "bottom": src.bounds.bottom,
            "right": src.bounds.right,
            "top": src.bounds.top
        }
        res_x, res_y = src.res
        height, width = src.height, src.width
        
    print(f"    Raster Dimensions: {height}x{width}, Bands: {raw_sar.shape[0]}, CRS: {crs_str}")
    
    # Identify valid SAR data pixels vs NoData (swath edge)
    valid_sar_mask = ~np.isnan(raw_sar[0]) & ~np.isnan(raw_sar[1])
    n_valid_sar = int(np.sum(valid_sar_mask))
    n_nodata_sar = int(np.sum(~valid_sar_mask))
    print(f"    Valid Data Pixels: {n_valid_sar:,} ({n_valid_sar / (height * width) * 100:.1f}%), NoData: {n_nodata_sar:,}")
    
    # Preprocess
    sar_norm = preprocess_sar(raw_sar)
    
    # Full tile sliding window inference
    print(f"[*] Running full-tile sliding window inference (stride={DEFAULT_STRIDE})...")
    t0 = time.time()
    full_prob = infer_full_tile_sliding_window(model, sar_norm, device, stride=DEFAULT_STRIDE)
    infer_duration_ms = (time.time() - t0) * 1000.0
    print(f"[OK] Inference complete in {infer_duration_ms:.1f} ms.")
    
    # Center crop inference for comparison
    center_prob = infer_center_crop(model, sar_norm, device)
    
    # Generate binary mask
    binary_mask = (full_prob >= threshold).astype(np.uint8)
    
    # Compute Geodesic Flood Area explicitly excluding NoData
    area_metrics = compute_geodesic_mask_area(
        mask=binary_mask,
        transform=transform_tuple,
        crs=crs_obj,
        valid_mask=valid_sar_mask
    )
    
    # Read ground truth label if provided
    gt_array = None
    full_tile_metrics = {}
    center_crop_metrics = {}
    
    if label_path and label_path.exists():
        print(f"[*] Reading reference label: {label_path}...")
        with rasterio.open(label_path) as lbl_src:
            gt_array = lbl_src.read(1)
            
        print("[*] Computing validation metrics against ground truth...")
        full_tile_metrics = compute_metrics(binary_mask, gt_array)
        
        # Center crop metrics
        cy_start = (height - CROP_SIZE) // 2
        cx_start = (width - CROP_SIZE) // 2
        gt_center = gt_array[cy_start:cy_start + CROP_SIZE, cx_start:cx_start + CROP_SIZE]
        binary_center = (center_prob >= threshold).astype(np.uint8)
        center_crop_metrics = compute_metrics(binary_center, gt_center)
        
        print("\n=== Validation Results (Full Tile @ Threshold 0.25) ===")
        print(f"    Flood IoU:    {full_tile_metrics.get('iou', 0.0):.4f}")
        print(f"    Precision:    {full_tile_metrics.get('precision', 0.0):.4f}")
        print(f"    Recall:       {full_tile_metrics.get('recall', 0.0):.4f}")
        print(f"    F1 Score:     {full_tile_metrics.get('f1_score', 0.0):.4f}")
        print(f"    Accuracy:     {full_tile_metrics.get('accuracy', 0.0):.4f}")
        print(f"    True Flood:   {full_tile_metrics.get('ground_truth_flood_pixels', 0):,} pixels")
        print(f"    Pred Flood:   {full_tile_metrics.get('predicted_flood_pixels', 0):,} pixels")
        
        # Also compute at default threshold 0.50 for standard reporting
        binary_mask_50 = (full_prob >= 0.50).astype(np.uint8)
        metrics_50 = compute_metrics(binary_mask_50, gt_array)
        print("\n=== Validation Results (Full Tile @ Threshold 0.50) ===")
        print(f"    Flood IoU:    {metrics_50.get('iou', 0.0):.4f}")
        print(f"    Precision:    {metrics_50.get('precision', 0.0):.4f}")
        print(f"    Recall:       {metrics_50.get('recall', 0.0):.4f}")
    
    # Save visualizations
    save_visualizations(
        output_dir=output_dir,
        raw_sar=raw_sar,
        full_prob=full_prob,
        binary_mask=binary_mask,
        threshold=threshold,
        gt=gt_array
    )
    
    # Write metadata JSON
    metadata = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "source_tile": {
            "path": str(sar_path),
            "filename": sar_path.name,
            "dimensions": {"height": height, "width": width, "bands": raw_sar.shape[0]},
            "crs": crs_str,
            "transform": list(transform_tuple),
            "bounds": bounds_dict,
            "pixel_resolution": {"res_x": res_x, "res_y": res_y}
        },
        "model": {
            "name": "Radiance SEN1Floods11 v2 (ViT-B MAE + ConvTranspose Head)",
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_size_bytes": checkpoint_path.stat().st_size,
            "license": "CC-BY-4.0",
            "source_repo": "https://huggingface.co/stratus-labs/radiance-sen1floods11-v1",
            "device": str(device),
            "device_name": torch.cuda.get_device_name(0) if device.type == "cuda" else "Host CPU"
        },
        "preprocessing": {
            "nan_handling": "np.nan_to_num(nan=0.0)",
            "clipping_db": [DB_CLIP_MIN, DB_CLIP_MAX],
            "ssl4eo_s12_normalization": {
                "mean": SSL4EO_S12_MEAN.tolist(),
                "std": SSL4EO_S12_STD.tolist()
            }
        },
        "inference_configuration": {
            "protocol": "full_tile_sliding_window",
            "window_size": CROP_SIZE,
            "stride": DEFAULT_STRIDE,
            "operating_threshold": threshold,
            "standard_threshold": 0.50,
            "inference_duration_ms": round(infer_duration_ms, 2),
            "total_execution_seconds": round(time.time() - start_time, 2)
        },
        "flood_extent_summary": {
            "total_raster_pixels": height * width,
            "valid_data_pixels": n_valid_sar,
            "nodata_pixels": n_nodata_sar,
            "predicted_flood_pixels": area_metrics["active_flood_pixels"],
            "flood_fraction_of_valid_data": round(area_metrics["active_flood_pixels"] / max(n_valid_sar, 1), 4),
            "flood_fraction_of_total_tile": round(area_metrics["active_flood_pixels"] / (height * width), 4),
            "estimated_flood_area_km2": area_metrics["estimated_flood_area_km2"],
            "estimated_flood_area_m2": area_metrics["estimated_flood_area_m2"],
            "mean_pixel_area_m2": area_metrics["mean_pixel_area_m2"],
            "area_calculation_method": area_metrics["calculation_method"],
            "is_ground_truth_area": False
        },
        "validation_metrics": {
            "reference_label_file": str(label_path.name) if label_path else None,
            "full_tile_operating_threshold_0_25": full_tile_metrics,
            "full_tile_standard_threshold_0_50": metrics_50 if label_path else None,
            "center_crop_threshold_0_25": center_crop_metrics
        }
    }
    
    metadata_path = output_dir / "inference_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)
        
    print(f"[OK] Saved inference metadata: {metadata_path}")
    print(f"    Estimated Flood Area: {area_metrics['estimated_flood_area_km2']} km^2 ({area_metrics['calculation_method']})")
    print(f"\n[OK] Pipeline completed successfully in {time.time() - start_time:.2f}s.")
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run FLOODLENS-AI Milestone 1 Inference on a Sentinel-1 Tile")
    parser.add_argument("--input", type=Path, default=DATA_SAMPLE_DIR / "Bolivia_103757_S1Hand.tif", help="Input S1 GeoTIFF")
    parser.add_argument("--label", type=Path, default=DATA_SAMPLE_DIR / "Bolivia_103757_LabelHand.tif", help="Ground Truth Label GeoTIFF")
    parser.add_argument("--checkpoint", type=Path, default=RADIANCE_SRC / "best.pt", help="Path to best.pt checkpoint")
    parser.add_argument("--output-dir", type=Path, default=BASE_DIR / "outputs", help="Output directory")
    parser.add_argument("--threshold", type=float, default=0.25, help="Operating threshold (default: 0.25)")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"], help="Execution device")
    
    args = parser.parse_args()
    run_inference_pipeline(
        sar_path=args.input,
        label_path=args.label,
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        threshold=args.threshold,
        device_name=args.device
    )
