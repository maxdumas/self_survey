"""
Contour generation from LiDAR ground points.

This module generates elevation contours from classified LiDAR point clouds.
The workflow:

1. Extract ground points (classification=2)
2. Interpolate to a regular grid (DEM)
3. Extract contour lines at specified intervals
4. Export to CAD-friendly formats (DXF)

Algorithm Details
-----------------

**Grid Interpolation**

We use scipy's griddata with linear interpolation to create a Digital Elevation
Model (DEM) from the irregular ground points. Linear interpolation is chosen over
cubic because:

- More robust to outliers and noise in LiDAR data
- Faster computation for large point clouds
- Less prone to overshooting between sparse points

The grid resolution determines contour smoothness. Default is 1 foot, which
provides good detail for property-scale surveys without excessive computation.

**Contour Extraction**

Matplotlib's contour algorithm (based on marching squares) extracts isolines
at specified elevation intervals. The algorithm:

1. Walks the grid finding cells that cross each elevation
2. Interpolates the crossing point within each cell
3. Connects crossings into continuous polylines

**Coordinate Handling**

All coordinates are preserved in the source CRS (typically State Plane feet).
The DXF output maintains these coordinates so the contours align with other
survey data in the same coordinate system.
"""

from typing import Any

import numpy as np
from numpy.typing import NDArray


def extract_ground_points(
    las: Any,
    classification: int = 2,
) -> NDArray[np.float64]:
    """
    Extract ground-classified points from a LAS file.

    Parameters
    ----------
    las : laspy.LasData
        Input LAS data
    classification : int
        Classification code for ground (default: 2 per ASPRS standard)

    Returns
    -------
    points : ndarray of shape (N, 3)
        Ground points as (x, y, z) array
    """
    mask = las.classification == classification
    x = np.array(las.x[mask])
    y = np.array(las.y[mask])
    z = np.array(las.z[mask])
    return np.column_stack((x, y, z))


def create_dem(
    points: NDArray[np.float64],
    resolution: float = 1.0,
    padding: float = 0.0,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """
    Create a Digital Elevation Model (DEM) from ground points.

    Interpolates irregular ground points onto a regular grid using
    linear interpolation.

    Parameters
    ----------
    points : ndarray of shape (N, 3)
        Ground points as (x, y, z) array
    resolution : float
        Grid cell size in the same units as input coordinates
    padding : float
        Extra padding around data extent

    Returns
    -------
    grid_x : ndarray of shape (ny, nx)
        X coordinates of grid points
    grid_y : ndarray of shape (ny, nx)
        Y coordinates of grid points
    grid_z : ndarray of shape (ny, nx)
        Interpolated elevations (NaN where extrapolation would occur)
    """
    from scipy.interpolate import griddata

    x, y, z = points[:, 0], points[:, 1], points[:, 2]

    # Define grid extent
    x_min, x_max = x.min() - padding, x.max() + padding
    y_min, y_max = y.min() - padding, y.max() + padding

    # Create regular grid
    grid_x_1d = np.arange(x_min, x_max + resolution, resolution)
    grid_y_1d = np.arange(y_min, y_max + resolution, resolution)
    grid_x, grid_y = np.meshgrid(grid_x_1d, grid_y_1d)

    # Interpolate
    grid_z = griddata(
        points=(x, y),
        values=z,
        xi=(grid_x, grid_y),
        method="linear",
        fill_value=np.nan,
    )

    return grid_x, grid_y, grid_z


def generate_contours(
    grid_x: NDArray[np.float64],
    grid_y: NDArray[np.float64],
    grid_z: NDArray[np.float64],
    interval: float = 2.0,
    base_elevation: float | None = None,
) -> list[tuple[float, list[NDArray[np.float64]]]]:
    """
    Generate contour lines from a DEM grid.

    Parameters
    ----------
    grid_x, grid_y, grid_z : ndarray
        DEM grid from create_dem()
    interval : float
        Elevation interval between contours (e.g., 2.0 for 2-foot contours)
    base_elevation : float, optional
        Base elevation for contour levels. If None, uses the minimum
        grid elevation rounded down to the nearest interval.

    Returns
    -------
    contours : list of (elevation, polylines)
        Each entry is (elevation, list of polyline arrays).
        Each polyline is an ndarray of shape (M, 2) with (x, y) coordinates.
    """
    import matplotlib.pyplot as plt

    # Determine contour levels
    z_min = np.nanmin(grid_z)
    z_max = np.nanmax(grid_z)

    if base_elevation is None:
        base_elevation = np.floor(z_min / interval) * interval

    levels = np.arange(base_elevation, z_max + interval, interval)
    # Filter to levels within data range (with small buffer)
    levels = levels[(levels >= z_min - interval) & (levels <= z_max + interval)]

    # Generate contours using matplotlib (doesn't display, just computes)
    fig, ax = plt.subplots()
    cs = ax.contour(grid_x, grid_y, grid_z, levels=levels)
    plt.close(fig)

    # Extract contour paths
    # Note: matplotlib 3.8+ deprecated cs.collections, use cs.allsegs instead
    contours = []
    for i, level in enumerate(cs.levels):
        polylines = []
        # cs.allsegs[i] is a list of (N, 2) arrays for each segment at this level
        for segment in cs.allsegs[i]:
            if len(segment) > 1:
                polylines.append(np.array(segment))
        if polylines:
            contours.append((float(level), polylines))

    return contours


def export_to_dxf(
    contours: list[tuple[float, list[NDArray[np.float64]]]],
    output_path: str,
    layer_prefix: str = "CONTOUR",
    index_interval: int = 5,
    major_layer: str | None = None,
    minor_layer: str | None = None,
    boundary_polygon: NDArray[np.float64] | None = None,
) -> dict[str, Any]:
    """
    Export contours to DXF format.

    Creates a DXF file with contour polylines. Contours are organized into
    layers for major (index) and minor contours, following CAD conventions.

    Parameters
    ----------
    contours : list of (elevation, polylines)
        Output from generate_contours()
    output_path : str
        Output DXF file path
    layer_prefix : str
        Prefix for layer names
    index_interval : int
        Every Nth contour is an index (major) contour.
        E.g., with 2-foot interval and index_interval=5, every 10-foot
        contour is major.
    major_layer : str, optional
        Layer name for major contours. Default: "{prefix}_MAJOR"
    minor_layer : str, optional
        Layer name for minor contours. Default: "{prefix}_MINOR"
    boundary_polygon : ndarray, optional
        Nx2 array of (x, y) coordinates forming the iPhone scan boundary.
        If provided, renders as a filled HATCH on the IPHONE_BOUNDARY layer.

    Returns
    -------
    stats : dict
        Statistics about the export (contour counts, elevation range, etc.)
    """
    import ezdxf
    from ezdxf import units

    # Create DXF document
    doc = ezdxf.new("R2010")  # AutoCAD 2010 format for broad compatibility
    doc.units = units.FT  # Set units to feet
    msp = doc.modelspace()

    # Set up layers
    if major_layer is None:
        major_layer = f"{layer_prefix}_MAJOR"
    if minor_layer is None:
        minor_layer = f"{layer_prefix}_MINOR"

    # Create layers with appropriate colors
    # Major contours: red (1), heavier lineweight
    doc.layers.add(major_layer, color=1, lineweight=35)
    # Minor contours: green (3), lighter lineweight
    doc.layers.add(minor_layer, color=3, lineweight=18)

    # Track statistics
    stats = {
        "major_contours": 0,
        "minor_contours": 0,
        "total_polylines": 0,
        "total_vertices": 0,
        "elevation_min": float("inf"),
        "elevation_max": float("-inf"),
    }

    for i, (elevation, polylines) in enumerate(contours):
        # Determine if this is a major (index) contour
        is_major = (i % index_interval) == 0 if index_interval > 0 else False
        layer = major_layer if is_major else minor_layer

        stats["elevation_min"] = min(stats["elevation_min"], elevation)
        stats["elevation_max"] = max(stats["elevation_max"], elevation)

        for polyline in polylines:
            if len(polyline) < 2:
                continue

            # Create true 3D polyline for Revit compatibility
            points_3d = [(x, y, elevation) for x, y in polyline]
            msp.add_polyline3d(
                points_3d,
                dxfattribs={"layer": layer},
            )

            stats["total_polylines"] += 1
            stats["total_vertices"] += len(polyline)

        if is_major:
            stats["major_contours"] += 1
        else:
            stats["minor_contours"] += 1

    # Add iPhone scan boundary if provided
    if boundary_polygon is not None and len(boundary_polygon) >= 3:
        # Create boundary layer (light green, dashed outline)
        doc.layers.add("IPHONE_BOUNDARY", color=3)  # Green

        # Add filled HATCH for the boundary region
        hatch = msp.add_hatch(color=3)  # Green fill
        hatch.set_pattern_fill("SOLID")
        hatch.dxf.layer = "IPHONE_BOUNDARY"

        # Add the polygon path (must be closed)
        hatch.paths.add_polyline_path(
            [(x, y) for x, y in boundary_polygon],
            is_closed=True,
        )

        # Also add outline polyline for visibility
        msp.add_lwpolyline(
            [(x, y) for x, y in boundary_polygon],
            close=True,
            dxfattribs={
                "layer": "IPHONE_BOUNDARY",
                "color": 3,
            },
        )

        stats["has_boundary"] = True
        stats["boundary_vertices"] = len(boundary_polygon)

    # Save DXF
    doc.saveas(output_path)

    return stats


def export_to_shapefile(
    contours: list[tuple[float, list[NDArray[np.float64]]]],
    output_path: str,
    crs_wkt: str | None = None,
) -> dict[str, Any]:
    """
    Export contours to Shapefile format.

    Creates a shapefile with contour polylines and elevation attributes.

    Parameters
    ----------
    contours : list of (elevation, polylines)
        Output from generate_contours()
    output_path : str
        Output shapefile path (should end in .shp)
    crs_wkt : str, optional
        Coordinate reference system as WKT string for the .prj file

    Returns
    -------
    stats : dict
        Statistics about the export
    """
    import json
    from pathlib import Path

    # We'll use a simple approach without requiring fiona/geopandas
    # by writing GeoJSON and optionally converting
    # For now, write as GeoJSON which is widely supported

    output_path = Path(output_path)
    if output_path.suffix.lower() == ".shp":
        # Redirect to GeoJSON
        output_path = output_path.with_suffix(".geojson")
        print(f"  Note: Writing as GeoJSON to {output_path}")

    features = []
    stats = {
        "total_contours": 0,
        "total_polylines": 0,
        "elevation_min": float("inf"),
        "elevation_max": float("-inf"),
    }

    for elevation, polylines in contours:
        stats["elevation_min"] = min(stats["elevation_min"], elevation)
        stats["elevation_max"] = max(stats["elevation_max"], elevation)
        stats["total_contours"] += 1

        for polyline in polylines:
            if len(polyline) < 2:
                continue

            feature = {
                "type": "Feature",
                "properties": {
                    "elevation": elevation,
                },
                "geometry": {
                    "type": "LineString",
                    "coordinates": [[float(x), float(y)] for x, y in polyline],
                },
            }
            features.append(feature)
            stats["total_polylines"] += 1

    geojson = {
        "type": "FeatureCollection",
        "features": features,
    }

    if crs_wkt:
        # Add CRS info (non-standard but useful)
        geojson["crs"] = {
            "type": "name",
            "properties": {"name": crs_wkt[:100] + "..."},
        }

    with open(output_path, "w") as f:
        json.dump(geojson, f)

    return stats
