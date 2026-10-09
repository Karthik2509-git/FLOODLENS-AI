"""
Unit tests for FLOODLENS-AI Geospatial Area Computation
Verifies geodesic ellipsoidal integration across geographic, projected, and nodata rasters.
"""

import sys
import unittest
import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_origin
from pathlib import Path

# Add scripts directory to path
BASE_DIR = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = BASE_DIR / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from infer_tile import compute_geodesic_mask_area


class TestGeospatialAreaCalculation(unittest.TestCase):

    def test_projected_crs_planar_area(self):
        """Test exact planar area calculation for projected CRS (e.g., UTM in meters)."""
        H, W = 100, 100
        # 10m x 10m pixels
        transform = from_origin(500000.0, 4000000.0, 10.0, 10.0)
        crs = CRS.from_epsg(32620)  # UTM Zone 20N
        
        # All 10,000 pixels active
        mask = np.ones((H, W), dtype=np.uint8)
        
        res = compute_geodesic_mask_area(mask, transform, crs)
        
        # 10,000 pixels * 100 m^2 = 1,000,000 m^2 = 1.0 km^2
        self.assertEqual(res["active_flood_pixels"], 10000)
        self.assertAlmostEqual(res["estimated_flood_area_m2"], 1000000.0, places=1)
        self.assertAlmostEqual(res["estimated_flood_area_km2"], 1.0, places=4)
        self.assertAlmostEqual(res["mean_pixel_area_m2"], 100.0, places=2)
        self.assertIn("Projected planar", res["calculation_method"])

    def test_geographic_crs_equator_vs_high_latitude(self):
        """Test WGS84 geodesic latitude convergence at Equator vs 60 degrees North."""
        H, W = 50, 50
        pixel_deg = 0.0001  # ~11.1m lat, ~11.1m lon at equator
        
        # 1. Equator (0 deg lat)
        transform_eq = from_origin(0.0, 0.0025, pixel_deg, pixel_deg)
        crs_geo = CRS.from_epsg(4326)
        mask = np.ones((H, W), dtype=np.uint8)
        
        res_eq = compute_geodesic_mask_area(mask, transform_eq, crs_geo)
        
        # 2. High Latitude (60 deg North) where cos(60) = 0.5 (longitude scale is halved)
        transform_60n = from_origin(0.0, 60.0025, pixel_deg, pixel_deg)
        res_60n = compute_geodesic_mask_area(mask, transform_60n, crs_geo)
        
        # Ratio of 60N pixel area to Equator pixel area should be approximately cos(60) = 0.50
        ratio = res_60n["total_area_m2"] if "total_area_m2" in res_60n else res_60n["estimated_flood_area_m2"]
        ratio = ratio / (res_eq["estimated_flood_area_m2"])
        
        self.assertAlmostEqual(ratio, 0.50, delta=0.015)
        self.assertIn("WGS84 ellipsoidal", res_eq["calculation_method"])

    def test_nodata_exclusion(self):
        """Test that NoData / invalid pixels are strictly excluded from area calculation."""
        H, W = 100, 100
        transform = from_origin(500000.0, 4000000.0, 10.0, 10.0)
        crs = CRS.from_epsg(32620)
        
        mask = np.ones((H, W), dtype=np.uint8)  # 10,000 candidate flood pixels
        valid_mask = np.zeros((H, W), dtype=bool)
        valid_mask[:50, :] = True  # Only top 50 rows (5,000 pixels) are valid data
        
        res = compute_geodesic_mask_area(mask, transform, crs, valid_mask=valid_mask)
        
        # Exactly 5,000 pixels active -> 500,000 m^2 = 0.5 km^2
        self.assertEqual(res["active_flood_pixels"], 5000)
        self.assertAlmostEqual(res["estimated_flood_area_m2"], 500000.0, places=1)
        self.assertAlmostEqual(res["estimated_flood_area_km2"], 0.5, places=4)

    def test_real_sample_tile_geodesic_bounds(self):
        """Test geodesic area computation on real Bolivia Sentinel-1 test raster."""
        sample_path = BASE_DIR / "data" / "sample" / "Bolivia_103757_S1Hand.tif"
        if not sample_path.exists():
            self.skipTest(f"Sample raster not found at {sample_path}")
            
        with rasterio.open(sample_path) as src:
            raw = src.read()
            valid_sar = ~np.isnan(raw[0]) & ~np.isnan(raw[1])
            transform = src.transform
            crs = src.crs
            
        # Create test flood mask (e.g. entire valid area)
        test_mask = valid_sar.astype(np.uint8)
        res = compute_geodesic_mask_area(test_mask, transform, crs, valid_mask=valid_sar)
        
        # Valid pixels: 101,623
        self.assertEqual(res["active_flood_pixels"], 101623)
        # Mean pixel area in Bolivia (-14.2 deg) should be ~96.36 m^2
        self.assertAlmostEqual(res["mean_pixel_area_m2"], 96.36, delta=0.5)
        # Total valid area: 101,623 * 96.36 m^2 ~ 9.79 km^2
        self.assertAlmostEqual(res["estimated_flood_area_km2"], 9.79, delta=0.1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
