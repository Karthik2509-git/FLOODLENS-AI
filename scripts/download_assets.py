"""
Download script for FLOODLENS-AI Milestone 1 assets:
- Model checkpoint & model definition from HuggingFace
- Sample Sentinel-1 SAR GeoTIFF & ground truth label from Sen1Floods11
"""

import os
import sys
import urllib.request
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models" / "radiance"
DATA_RAW_DIR = BASE_DIR / "data" / "raw"
DATA_SAMPLE_DIR = BASE_DIR / "data" / "sample"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
DATA_SAMPLE_DIR.mkdir(parents=True, exist_ok=True)

DOWNLOADS = [
    {
        "url": "https://huggingface.co/stratus-labs/radiance-sen1floods11-v1/raw/main/model.py",
        "dest": MODELS_DIR / "model.py",
        "desc": "Radiance model architecture definition"
    },
    {
        "url": "https://huggingface.co/stratus-labs/radiance-sen1floods11-v1/resolve/main/best.pt",
        "dest": MODELS_DIR / "best.pt",
        "desc": "Radiance SEN1Floods11 v2 checkpoint (best.pt)"
    },
    {
        "url": "https://storage.googleapis.com/sen1floods11/v1.1/data/flood_events/HandLabeled/S1Hand/Bolivia_103757_S1Hand.tif",
        "dest": DATA_SAMPLE_DIR / "Bolivia_103757_S1Hand.tif",
        "desc": "Sample Sentinel-1 SAR GeoTIFF (Bolivia_103757)"
    },
    {
        "url": "https://storage.googleapis.com/sen1floods11/v1.1/data/flood_events/HandLabeled/LabelHand/Bolivia_103757_LabelHand.tif",
        "dest": DATA_SAMPLE_DIR / "Bolivia_103757_LabelHand.tif",
        "desc": "Sample Ground Truth Flood Label GeoTIFF (Bolivia_103757)"
    }
]

def download_with_progress(url: str, dest_path: Path, desc: str):
    if dest_path.exists() and dest_path.stat().st_size > 0:
        print(f"[*] Already exists: {dest_path.name} ({dest_path.stat().st_size:,} bytes) - Skipping.")
        return

    print(f"\n[+] Downloading {desc}...")
    print(f"    Source: {url}")
    print(f"    Target: {dest_path}")

    start_time = time.time()
    temp_dest = dest_path.with_suffix(dest_path.suffix + ".tmp")
    
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "FLOODLENS-AI-Downloader/1.0"}
    )

    with urllib.request.urlopen(req) as response, open(temp_dest, "wb") as out_file:
        total_size = int(response.headers.get("content-length", 0))
        downloaded = 0
        block_size = 1024 * 1024  # 1 MB

        while True:
            buffer = response.read(block_size)
            if not buffer:
                break
            downloaded += len(buffer)
            out_file.write(buffer)
            
            if total_size > 0:
                percent = downloaded * 100 / total_size
                mb_down = downloaded / (1024 * 1024)
                mb_total = total_size / (1024 * 1024)
                speed = mb_down / max(time.time() - start_time, 0.001)
                sys.stdout.write(f"\r    Progress: {percent:5.1f}% ({mb_down:6.1f} / {mb_total:6.1f} MB) @ {speed:5.1f} MB/s")
                sys.stdout.flush()

    if temp_dest.exists():
        temp_dest.replace(dest_path)
    
    elapsed = time.time() - start_time
    final_size = dest_path.stat().st_size
    print(f"\n    [OK] Done: {final_size:,} bytes in {elapsed:.1f}s")

if __name__ == "__main__":
    print("=== FLOODLENS-AI Asset Downloader ===")
    for item in DOWNLOADS:
        download_with_progress(item["url"], item["dest"], item["desc"])
    print("\n[OK] All required assets are in place.")
