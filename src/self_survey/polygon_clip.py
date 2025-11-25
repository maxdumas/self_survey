"""
Polygon clipping utilities for point clouds.

This module provides functions to clip point clouds to polygon boundaries
using efficient point-in-polygon tests.
"""

import numpy as np
from numpy.typing import NDArray

__all__ = ["points_in_polygon", "clip_to_polygon"]


def points_in_polygon(
    points: NDArray[np.floating],
    polygon: NDArray[np.floating],
) -> NDArray[np.bool_]:
    """
    Test which points are inside a polygon using ray casting.

    Uses the ray casting algorithm (also known as crossing number or
    even-odd rule) to determine point containment.

    Args:
        points: Nx2 array of (x, y) point coordinates
        polygon: Mx2 array of (x, y) polygon vertices (closed ring)

    Returns:
        Boolean array of length N, True if point is inside polygon
    """
    n_points = len(points)
    n_vertices = len(polygon)

    if n_vertices < 3:
        return np.zeros(n_points, dtype=bool)

    # Extract x and y coordinates
    px = points[:, 0]
    py = points[:, 1]

    # Initialize result array
    inside = np.zeros(n_points, dtype=bool)

    # Ray casting algorithm
    # For each edge of the polygon, count crossings
    j = n_vertices - 1
    for i in range(n_vertices):
        xi, yi = polygon[i]
        xj, yj = polygon[j]

        # Check if the horizontal ray from point crosses this edge
        # Edge goes from (xj, yj) to (xi, yi)

        # Condition: point's y is between edge's y values
        cond1 = (yi > py) != (yj > py)

        # Condition: point's x is to the left of the edge at point's y
        # x-intersection of edge at py: xj + (py - yj) * (xi - xj) / (yi - yj)
        with np.errstate(divide="ignore", invalid="ignore"):
            x_intersect = xj + (py - yj) * (xi - xj) / (yi - yj)

        cond2 = px < x_intersect

        # Toggle inside flag where both conditions are met
        inside = inside ^ (cond1 & cond2)

        j = i

    return inside


def clip_to_polygon(
    points: NDArray[np.floating],
    polygon: NDArray[np.floating],
    return_mask: bool = False,
) -> NDArray[np.floating] | tuple[NDArray[np.floating], NDArray[np.bool_]]:
    """
    Clip points to a polygon boundary.

    Args:
        points: Nx2 or Nx3 array of point coordinates (x, y) or (x, y, z)
        polygon: Mx2 array of (x, y) polygon vertices
        return_mask: If True, also return the boolean mask

    Returns:
        Clipped points array, and optionally the boolean mask
    """
    # Use only x, y for containment test
    xy_points = points[:, :2] if points.shape[1] > 2 else points

    mask = points_in_polygon(xy_points, polygon)
    clipped = points[mask]

    if return_mask:
        return clipped, mask
    return clipped


def polygon_area(polygon: NDArray[np.floating]) -> float:
    """
    Calculate the area of a polygon using the Shoelace formula.

    Args:
        polygon: Mx2 array of (x, y) polygon vertices

    Returns:
        Absolute area of the polygon
    """
    n = len(polygon)
    if n < 3:
        return 0.0

    # Shoelace formula
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += polygon[i, 0] * polygon[j, 1]
        area -= polygon[j, 0] * polygon[i, 1]

    return abs(area) / 2.0


def polygon_centroid(polygon: NDArray[np.floating]) -> tuple[float, float]:
    """
    Calculate the centroid of a polygon.

    Args:
        polygon: Mx2 array of (x, y) polygon vertices

    Returns:
        (x, y) centroid coordinates
    """
    n = len(polygon)
    if n < 3:
        return (0.0, 0.0)

    # Calculate centroid using the formula for polygon centroid
    cx = 0.0
    cy = 0.0
    signed_area = 0.0

    for i in range(n):
        j = (i + 1) % n
        cross = polygon[i, 0] * polygon[j, 1] - polygon[j, 0] * polygon[i, 1]
        signed_area += cross
        cx += (polygon[i, 0] + polygon[j, 0]) * cross
        cy += (polygon[i, 1] + polygon[j, 1]) * cross

    signed_area /= 2.0

    if abs(signed_area) < 1e-10:
        # Degenerate polygon, return mean of vertices
        return (float(polygon[:, 0].mean()), float(polygon[:, 1].mean()))

    cx /= 6.0 * signed_area
    cy /= 6.0 * signed_area

    return (cx, cy)


def buffer_polygon(
    polygon: NDArray[np.floating],
    distance: float,
) -> NDArray[np.floating]:
    """
    Create a simple buffer around a polygon by scaling from centroid.

    This is a simplified buffer that works by scaling the polygon
    outward (positive distance) or inward (negative distance) from
    its centroid. For more accurate buffering, use a GIS library.

    Args:
        polygon: Mx2 array of (x, y) polygon vertices
        distance: Buffer distance (positive = expand, negative = contract)

    Returns:
        Buffered polygon vertices
    """
    if len(polygon) < 3:
        return polygon.copy()

    # Calculate centroid
    cx, cy = polygon_centroid(polygon)

    # Calculate average distance from centroid
    distances = np.sqrt((polygon[:, 0] - cx) ** 2 + (polygon[:, 1] - cy) ** 2)
    avg_dist = distances.mean()

    if avg_dist < 1e-10:
        return polygon.copy()

    # Scale factor
    scale = (avg_dist + distance) / avg_dist

    # Scale polygon from centroid
    buffered = np.zeros_like(polygon)
    buffered[:, 0] = cx + (polygon[:, 0] - cx) * scale
    buffered[:, 1] = cy + (polygon[:, 1] - cy) * scale

    return buffered
