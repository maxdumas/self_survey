"""
Merge LiDAR LAS/LAZ tiles into a single point cloud.

This module provides functions for loading, merging, and saving LiDAR
point clouds while preserving metadata like classification and intensity.
"""

from pathlib import Path

import laspy
import numpy as np
import open3d as o3d

__all__ = [
    "load_las_to_open3d",
    "merge_point_clouds",
    "save_as_laz",
    "save_as_ply",
    "visualize",
]


def load_las_to_open3d(filepath: str) -> tuple[o3d.geometry.PointCloud, dict]:
    """
    Load a LAS/LAZ file into an Open3D point cloud.

    Returns the point cloud and metadata dict with classification,
    intensity, and other attributes for later reconstruction.
    """
    print(f"Loading {filepath}...")
    las = laspy.read(filepath)

    # Extract XYZ coordinates
    points = np.vstack((las.x, las.y, las.z)).T
    print(f"  Points: {len(points):,}")
    print(f"  Bounds X: [{las.x.min():.2f}, {las.x.max():.2f}]")
    print(f"  Bounds Y: [{las.y.min():.2f}, {las.y.max():.2f}]")
    print(f"  Bounds Z: [{las.z.min():.2f}, {las.z.max():.2f}]")

    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # Extract colors if present (RGB)
    if hasattr(las, "red") and hasattr(las, "green") and hasattr(las, "blue"):
        # LAS stores RGB as 16-bit, normalize to 0-1
        colors = np.vstack(
            (las.red / 65535.0, las.green / 65535.0, las.blue / 65535.0)
        ).T
        pcd.colors = o3d.utility.Vector3dVector(colors)
        print("  RGB colors: yes")
    else:
        print("  RGB colors: no")

    # Preserve other attributes for LAS export
    metadata = {
        "classification": np.array(las.classification),
        "intensity": np.array(las.intensity),
        "return_number": (
            np.array(las.return_number) if hasattr(las, "return_number") else None
        ),
        "number_of_returns": (
            np.array(las.number_of_returns)
            if hasattr(las, "number_of_returns")
            else None
        ),
        "header": las.header,
    }

    return pcd, metadata


def merge_point_clouds(
    pcd1: o3d.geometry.PointCloud,
    meta1: dict,
    pcd2: o3d.geometry.PointCloud,
    meta2: dict,
) -> tuple[o3d.geometry.PointCloud, dict]:
    """
    Merge two point clouds and their metadata.
    """
    print("\nMerging point clouds...")

    # Combine points
    merged = pcd1 + pcd2  # Open3D overloads + for point cloud concatenation

    print(f"  Total points: {len(merged.points):,}")

    # Merge metadata arrays
    merged_meta = {
        "classification": np.concatenate(
            [meta1["classification"], meta2["classification"]]
        ),
        "intensity": np.concatenate([meta1["intensity"], meta2["intensity"]]),
    }

    if meta1["return_number"] is not None and meta2["return_number"] is not None:
        merged_meta["return_number"] = np.concatenate(
            [meta1["return_number"], meta2["return_number"]]
        )
        merged_meta["number_of_returns"] = np.concatenate(
            [meta1["number_of_returns"], meta2["number_of_returns"]]
        )

    # Use header from first file as template
    merged_meta["header"] = meta1["header"]

    return merged, merged_meta


def save_as_laz(
    pcd: o3d.geometry.PointCloud,
    metadata: dict,
    output_path: str,
    compress: bool = True,
):
    """
    Save Open3D point cloud back to LAS/LAZ format with attributes.
    """
    print(f"\nSaving to {output_path}...")

    points = np.asarray(pcd.points)

    # Create new LAS file with same point format as source
    header = laspy.LasHeader(
        point_format=metadata["header"].point_format, version="1.4"
    )
    header.offsets = np.min(points, axis=0)
    header.scales = [0.001, 0.001, 0.001]  # 1mm precision

    # Copy CRS if present
    for vlr in metadata["header"].vlrs:
        header.vlrs.append(vlr)

    las = laspy.LasData(header)

    # Set coordinates
    las.x = points[:, 0]
    las.y = points[:, 1]
    las.z = points[:, 2]

    # Set attributes
    las.classification = metadata["classification"]
    las.intensity = metadata["intensity"]

    if metadata.get("return_number") is not None:
        las.return_number = metadata["return_number"]
        las.number_of_returns = metadata["number_of_returns"]

    # Set colors if present
    if pcd.has_colors():
        colors = np.asarray(pcd.colors)
        las.red = (colors[:, 0] * 65535).astype(np.uint16)
        las.green = (colors[:, 1] * 65535).astype(np.uint16)
        las.blue = (colors[:, 2] * 65535).astype(np.uint16)

    las.write(output_path)

    file_size = Path(output_path).stat().st_size / (1024 * 1024)
    print(f"  File size: {file_size:.1f} MB")


def save_as_ply(pcd: o3d.geometry.PointCloud, output_path: str):
    """
    Save as PLY (simpler, but loses LAS-specific attributes).
    """
    print(f"\nSaving to {output_path}...")
    o3d.io.write_point_cloud(output_path, pcd, write_ascii=False)
    file_size = Path(output_path).stat().st_size / (1024 * 1024)
    print(f"  File size: {file_size:.1f} MB")


def visualize(pcd: o3d.geometry.PointCloud, metadata: dict, color_by: str = "rgb"):
    """
    Quick visualization of the merged cloud.

    color_by: "rgb", "classification", "intensity", or "elevation"
    """
    print(f"\nVisualizing (colored by {color_by})...")

    vis_pcd = o3d.geometry.PointCloud(pcd)  # Copy to avoid modifying original

    if color_by == "classification":
        # Color by classification (ground=brown, vegetation=green, etc.)
        class_colors = {
            2: [0.6, 0.4, 0.2],  # Ground - brown
            3: [0.2, 0.6, 0.2],  # Low vegetation - light green
            4: [0.1, 0.5, 0.1],  # Medium vegetation - green
            5: [0.0, 0.4, 0.0],  # High vegetation - dark green
            6: [0.8, 0.2, 0.2],  # Building - red
            9: [0.3, 0.5, 0.9],  # Water - blue
        }
        colors = np.array(
            [class_colors.get(c, [0.5, 0.5, 0.5]) for c in metadata["classification"]]
        )
        vis_pcd.colors = o3d.utility.Vector3dVector(colors)

    elif color_by == "intensity":
        intensity = metadata["intensity"].astype(float)
        intensity = (intensity - intensity.min()) / (
            intensity.max() - intensity.min() + 1e-8
        )
        colors = np.column_stack([intensity, intensity, intensity])
        vis_pcd.colors = o3d.utility.Vector3dVector(colors)

    elif color_by == "elevation":
        points = np.asarray(pcd.points)
        z = points[:, 2]
        z_norm = (z - z.min()) / (z.max() - z.min() + 1e-8)
        # Blue (low) to red (high) colormap
        colors = np.column_stack([z_norm, 0.3 * np.ones_like(z_norm), 1 - z_norm])
        vis_pcd.colors = o3d.utility.Vector3dVector(colors)

    # RGB is already set if available

    o3d.visualization.draw_geometries(
        [vis_pcd],
        window_name="Merged LiDAR Tiles",
        width=1400,
        height=900,
        point_show_normal=False,
    )
