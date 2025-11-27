"""
Colorize a point cloud from orthoimagery.

This module provides functions to drape RGB colors from orthoimagery
onto LiDAR point clouds by sampling pixel values at each point location.
"""

import laspy
import numpy as np
import open3d as o3d
import rasterio
from pyproj import CRS, Transformer

__all__ = ["colorize_point_cloud", "colorize_from_ortho"]


def colorize_point_cloud(
    pcd: o3d.geometry.PointCloud,
    ortho_path: str,
    source_crs: CRS | None = None,
) -> o3d.geometry.PointCloud:
    """
    Add RGB colors to an Open3D point cloud by sampling from an orthophoto.

    Handles CRS reprojection automatically if source_crs is provided and
    differs from the orthoimagery CRS.

    Args:
        pcd: Open3D point cloud
        ortho_path: Path to orthoimagery file (GeoTIFF, JP2, etc.)
        source_crs: CRS of the point cloud coordinates. If None, assumes
            coordinates are already in the same CRS as the orthoimagery.

    Returns:
        The same point cloud with colors updated from the orthophoto
    """
    print(f"  Opening {ortho_path}...")

    points = np.asarray(pcd.points)
    xs = points[:, 0]
    ys = points[:, 1]

    with rasterio.open(ortho_path) as ortho:
        ortho_crs = CRS.from_user_input(ortho.crs) if ortho.crs else None

        # Reproject point coordinates if CRS differs
        if source_crs and ortho_crs and source_crs != ortho_crs:
            print(f"  Reprojecting points from {source_crs.name} to {ortho_crs.name}...")
            transformer = Transformer.from_crs(source_crs, ortho_crs, always_xy=True)
            xs, ys = transformer.transform(xs, ys)

        # Transform point coordinates to raster pixel coordinates
        rows, cols = rasterio.transform.rowcol(ortho.transform, xs, ys)
        rows = np.array(rows)
        cols = np.array(cols)

        # Check how many points fall within the image bounds
        valid_mask = (
            (rows >= 0) & (rows < ortho.height) &
            (cols >= 0) & (cols < ortho.width)
        )
        valid_count = np.sum(valid_mask)
        total_count = len(rows)

        if valid_count == 0:
            print(f"  Warning: No points fall within orthoimagery bounds")
            print(f"    Ortho bounds: {ortho.bounds}")
            print(f"    Point X range: {xs.min():.1f} - {xs.max():.1f}")
            print(f"    Point Y range: {ys.min():.1f} - {ys.max():.1f}")
            return pcd

        print(f"  Points within ortho bounds: {valid_count:,} of {total_count:,} ({100*valid_count/total_count:.1f}%)")

        # Read the full image
        print("  Reading orthoimagery...")
        rgb = ortho.read([1, 2, 3])  # Assumes bands 1,2,3 are RGB

        # Determine scale factor for normalization
        sample_max = max(rgb[0].max(), rgb[1].max(), rgb[2].max())
        if sample_max > 255:
            scale = 65535.0  # 16-bit imagery
        else:
            scale = 255.0  # 8-bit imagery

        # Sample colors ONLY for valid points (those within bounds)
        print("  Sampling colors...")
        valid_rows = rows[valid_mask]
        valid_cols = cols[valid_mask]

        red = rgb[0, valid_rows, valid_cols]
        green = rgb[1, valid_rows, valid_cols]
        blue = rgb[2, valid_rows, valid_cols]

    # Get or initialize colors array
    if pcd.has_colors():
        colors = np.asarray(pcd.colors)
    else:
        # Initialize with gray (0.5, 0.5, 0.5) for uncolored points
        colors = np.full((len(points), 3), 0.5, dtype=np.float64)

    # Update only the valid points with sampled colors
    colors[valid_mask, 0] = red.astype(np.float64) / scale
    colors[valid_mask, 1] = green.astype(np.float64) / scale
    colors[valid_mask, 2] = blue.astype(np.float64) / scale

    pcd.colors = o3d.utility.Vector3dVector(colors)
    print(f"  Colorized {valid_count:,} points (of {total_count:,} total)")

    return pcd


def colorize_from_ortho(
    las_path: str,
    ortho_path: str,
    output_path: str,
) -> laspy.LasData:
    """
    Add RGB colors to a LAS file by sampling from an orthophoto.

    Args:
        las_path: Path to input LAS/LAZ file
        ortho_path: Path to orthoimagery GeoTIFF file
        output_path: Path to write colorized LAS/LAZ file

    Returns:
        The colorized LAS data object
    """
    print(f"Loading {las_path}...")
    las = laspy.read(las_path)

    print(f"Opening {ortho_path}...")
    with rasterio.open(ortho_path) as ortho:
        # Get point coordinates
        xs = np.array(las.x)
        ys = np.array(las.y)

        # Transform point coordinates to raster pixel coordinates
        rows, cols = rasterio.transform.rowcol(ortho.transform, xs, ys)
        rows = np.array(rows)
        cols = np.array(cols)

        # Clamp to image bounds
        rows = np.clip(rows, 0, ortho.height - 1)
        cols = np.clip(cols, 0, ortho.width - 1)

        # Read the full image (or tile if huge)
        print("Reading orthoimagery...")
        rgb = ortho.read([1, 2, 3])  # Assumes bands 1,2,3 are RGB

        # Sample colors at each point
        print("Sampling colors...")
        red = rgb[0, rows, cols]
        green = rgb[1, rows, cols]
        blue = rgb[2, rows, cols]

    # Create new LAS with RGB
    print("Creating colorized point cloud...")

    # Need a point format that supports RGB (format 2, 3, 7, or 8)
    new_header = laspy.LasHeader(point_format=3, version="1.4")
    new_header.offsets = las.header.offsets
    new_header.scales = las.header.scales

    # Copy VLRs (CRS info)
    for vlr in las.header.vlrs:
        new_header.vlrs.append(vlr)

    new_las = laspy.LasData(new_header)
    new_las.x = las.x
    new_las.y = las.y
    new_las.z = las.z
    new_las.intensity = las.intensity
    new_las.classification = las.classification

    # LAS stores RGB as 16-bit
    new_las.red = red.astype(np.uint16) * 256
    new_las.green = green.astype(np.uint16) * 256
    new_las.blue = blue.astype(np.uint16) * 256

    print(f"Saving to {output_path}...")
    new_las.write(output_path)

    print("Done!")
    return new_las
