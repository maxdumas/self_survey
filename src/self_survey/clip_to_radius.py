"""
Clip a LAZ/LAS file to a radius around a latitude/longitude point.

This module provides functions for coordinate transformation and
spatial clipping of LiDAR point clouds.
"""

import laspy
import numpy as np
from pyproj import CRS, Transformer

__all__ = ["get_crs_from_las", "transform_latlon_to_crs", "clip_to_radius"]


def get_crs_from_las(las: laspy.LasData) -> CRS | None:
    """
    Extract CRS from LAS file VLRs.

    Args:
        las: A laspy LasData object

    Returns:
        A pyproj CRS object, or None if CRS cannot be determined
    """
    # Try the built-in parser first (works for many LAS 1.4 files)
    try:
        crs = las.header.parse_crs()
        if crs:
            return CRS.from_wkt(crs.to_wkt())
    except Exception:
        pass

    # Check VLRs manually for WKT or GeoTIFF keys
    for vlr in las.header.vlrs:
        # WKT stored in OGC WKT VLR
        if vlr.user_id == "LASF_Projection" and vlr.record_id == 2111:
            wkt = vlr.record_data.decode("utf-8", errors="ignore").strip("\x00")
            return CRS.from_wkt(wkt)

        # Also check for EPSG in description (some files store it there)
        if "EPSG" in str(vlr.description):
            try:
                epsg = int("".join(filter(str.isdigit, vlr.description)))
                return CRS.from_epsg(epsg)
            except Exception:
                pass

    return None


def transform_latlon_to_crs(
    lat: float,
    lon: float,
    target_crs: CRS,
) -> tuple[float, float]:
    """
    Transform a WGS84 lat/lon point to the target CRS.

    Args:
        lat: Latitude in decimal degrees (WGS84)
        lon: Longitude in decimal degrees (WGS84)
        target_crs: Target coordinate reference system

    Returns:
        Tuple of (x, y) in target CRS units
    """
    wgs84 = CRS.from_epsg(4326)
    transformer = Transformer.from_crs(wgs84, target_crs, always_xy=True)

    # Note: Transformer expects (lon, lat) order with always_xy=True
    x, y = transformer.transform(lon, lat)
    return x, y


def clip_to_radius(
    las: laspy.LasData,
    center_x: float,
    center_y: float,
    radius: float,
) -> tuple[laspy.LasData, int, int]:
    """
    Clip LAS points to within radius of center point.

    Args:
        las: Input LAS data
        center_x: X coordinate of center point (in file CRS units)
        center_y: Y coordinate of center point (in file CRS units)
        radius: Clip radius (in file CRS units)

    Returns:
        Tuple of (clipped_las, original_count, clipped_count)
    """
    points = np.vstack((las.x, las.y)).T
    original_count = len(points)

    # Calculate distances from center
    distances = np.sqrt((points[:, 0] - center_x) ** 2 + (points[:, 1] - center_y) ** 2)

    # Create mask for points within radius
    mask = distances <= radius
    clipped_count = np.sum(mask)

    # Create new LAS with only the points within radius
    clipped = las[mask]

    return clipped, original_count, clipped_count
