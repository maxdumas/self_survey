"""
Register and merge iPhone LiDAR scans with NYS reference data.

This module provides functions for:
- Loading and transforming point clouds to a common CRS
- ICP (Iterative Closest Point) alignment using ground points
- Transferring ground classification from reference to iPhone data
- Merging point clouds with replacement in overlap regions

Algorithmic Overview
--------------------

**Ground-to-All ICP Strategy**

iPhone scans from Polycam don't include ground classification. Instead of
requiring ground-to-ground alignment, we use a "ground-to-all" strategy:

- Target: Reference (NYS) GROUND points only (classification=2)
- Source: ALL iPhone points

This works because:
1. ICP's max_correspondence_distance naturally filters non-ground iPhone points
   (trees, vegetation won't find nearby ground correspondences)
2. The ground surface is the most reliable common feature between aerial
   LiDAR and iPhone scans, especially in woodland environments
3. Point-to-plane ICP is well-suited to ground surfaces

**Ground Classification Transfer**

After alignment, we infer ground classification for iPhone points by proximity
to the reference ground surface:

For each iPhone point (x, y, z):
    1. Find reference ground points within horizontal_tolerance of (x, y)
    2. If any have |z_ref - z_iphone| <= vertical_tolerance:
       → Classify as ground (code 2)

This preserves the professional ground classification from NYS data while
enabling consistent downstream processing of the merged point cloud.

**Merge Strategy**

iPhone data REPLACES (not blends with) NYS data in the overlap region.
This avoids double-density artifacts and provides clear data provenance.
"""

import json

import laspy
import numpy as np
import open3d as o3d
from pyproj import CRS, Transformer
from scipy.spatial import ConvexHull

from self_survey.clip_to_radius import get_crs_from_las

__all__ = [
    "load_and_transform_to_crs",
    "extract_ground_points",
    "icp_align",
    "apply_transform_to_las",
    "transfer_ground_classification",
    "merge_with_replacement",
    "get_iphone_boundary_from_las",
]


def load_and_transform_to_crs(
    las_path: str,
    target_crs: CRS,
    initial_position: tuple[float, float] | None = None,
    reference_las: "laspy.LasData | None" = None,
) -> tuple[laspy.LasData, CRS, np.ndarray | None]:
    """
    Load a LAS file and transform coordinates to target CRS if needed.

    For iPhone scans in local coordinates (no CRS metadata), the initial_position
    parameter specifies where the scan was taken. The local coordinates will be
    translated so that the scan's centroid is at initial_position.

    If reference_las is provided, the Z coordinates will also be adjusted to match
    the approximate ground elevation at the initial position.

    Args:
        las_path: Path to LAS/LAZ file
        target_crs: Target coordinate reference system
        initial_position: (x, y) in target CRS where the scan was taken.
            Required for scans without CRS metadata (local coordinates).
        reference_las: Reference LAS data for estimating ground elevation.
            Used to align Z coordinates for local-coordinate scans.

    Returns:
        Tuple of (las_data, source_crs, transformation_applied)
        transformation_applied is None if no transform was needed,
        otherwise it's the offset applied [dx, dy, dz]
    """
    print(f"Loading {las_path}...")
    las = laspy.read(las_path)
    source_crs = get_crs_from_las(las)

    if source_crs is None:
        print("  WARNING: Could not detect CRS from file")
        print("  File appears to be in local coordinates")

        # Calculate scan extent (for informational purposes)
        x_center = (las.x.max() + las.x.min()) / 2
        y_center = (las.y.max() + las.y.min()) / 2

        print(f"  Local bounds X: [{las.x.min():.2f}, {las.x.max():.2f}]")
        print(f"  Local bounds Y: [{las.y.min():.2f}, {las.y.max():.2f}]")
        print(f"  Local center: ({x_center:.2f}, {y_center:.2f})")

        if initial_position is None:
            raise ValueError(
                "iPhone scan has no CRS metadata and appears to be in local coordinates. "
                "Please provide --lat and --lon to specify the scan location."
            )

        # Translate local coordinates to target CRS
        # Center the scan on the initial_position
        target_x, target_y = initial_position
        offset_x = target_x - x_center
        offset_y = target_y - y_center

        # Estimate Z offset from reference ground elevation
        offset_z = 0.0
        if reference_las is not None:
            # Find reference ground points near the initial position
            ref_x = np.array(reference_las.x)
            ref_y = np.array(reference_las.y)
            ref_z = np.array(reference_las.z)
            ref_class = np.array(reference_las.classification)

            # Use ground points (classification=2) if available
            ground_mask = ref_class == 2
            if np.sum(ground_mask) > 0:
                ref_x_ground = ref_x[ground_mask]
                ref_y_ground = ref_y[ground_mask]
                ref_z_ground = ref_z[ground_mask]
            else:
                # Fall back to all points
                ref_x_ground = ref_x
                ref_y_ground = ref_y
                ref_z_ground = ref_z

            # Find points within a search radius of the initial position
            search_radius = 50.0  # feet
            distances = np.sqrt((ref_x_ground - target_x)**2 + (ref_y_ground - target_y)**2)
            nearby_mask = distances < search_radius

            if np.sum(nearby_mask) > 0:
                # Use median elevation of nearby ground points
                ref_ground_z = np.median(ref_z_ground[nearby_mask])
                # iPhone scan Z is likely relative to ground (0 = ground level)
                # So offset by the reference ground elevation
                iphone_min_z = las.z.min()
                offset_z = ref_ground_z - iphone_min_z
                print(f"  Reference ground elevation near center: {ref_ground_z:.2f}")
                print(f"  iPhone min Z: {iphone_min_z:.2f}")
            else:
                print(f"  WARNING: No reference points found within {search_radius} units of center")
                print(f"  Z offset will be 0 - vertical alignment may be wrong")

        print(f"\n  Translating to target CRS...")
        print(f"  Target center: ({target_x:.2f}, {target_y:.2f})")
        print(f"  Offset XY: ({offset_x:.2f}, {offset_y:.2f})")
        print(f"  Offset Z: {offset_z:.2f}")

        x_new = np.array(las.x) + offset_x
        y_new = np.array(las.y) + offset_y
        z_new = np.array(las.z) + offset_z

        # Create new LAS with translated coordinates
        new_header = laspy.LasHeader(point_format=las.header.point_format, version="1.4")
        new_header.scales = las.header.scales
        new_header.offsets = [np.min(x_new), np.min(y_new), np.min(z_new)]

        # Copy VLRs
        for vlr in las.header.vlrs:
            new_header.vlrs.append(vlr)

        new_las = laspy.LasData(new_header)
        new_las.x = x_new
        new_las.y = y_new
        new_las.z = z_new

        # Copy other attributes
        new_las.intensity = las.intensity
        new_las.classification = las.classification

        if hasattr(las, "red") and hasattr(las, "green") and hasattr(las, "blue"):
            new_las.red = las.red
            new_las.green = las.green
            new_las.blue = las.blue

        if hasattr(las, "return_number"):
            new_las.return_number = las.return_number
        if hasattr(las, "number_of_returns"):
            new_las.number_of_returns = las.number_of_returns

        offset = np.array([offset_x, offset_y, offset_z])

        print(f"  New bounds X: [{x_new.min():.2f}, {x_new.max():.2f}]")
        print(f"  New bounds Y: [{y_new.min():.2f}, {y_new.max():.2f}]")
        print(f"  New bounds Z: [{z_new.min():.2f}, {z_new.max():.2f}]")

        return new_las, target_crs, offset

    print(f"  Source CRS: {source_crs.name}")
    print(f"  Target CRS: {target_crs.name}")
    print(f"  Points: {len(las.points):,}")

    # Check if transformation is needed
    if source_crs.equals(target_crs):
        print("  CRS match - no transformation needed")
        return las, source_crs, None

    print("  Transforming coordinates...")

    # Create transformer
    transformer = Transformer.from_crs(source_crs, target_crs, always_xy=True)

    # Transform XY coordinates
    x_orig = np.array(las.x)
    y_orig = np.array(las.y)
    z_orig = np.array(las.z)

    x_new, y_new = transformer.transform(x_orig, y_orig)

    # Handle vertical datum transformation if needed
    # For simplicity, we keep Z as-is (both should be orthometric heights)
    z_new = z_orig

    # Create new LAS with transformed coordinates
    new_header = laspy.LasHeader(point_format=las.header.point_format, version="1.4")
    new_header.scales = las.header.scales
    new_header.offsets = [np.min(x_new), np.min(y_new), np.min(z_new)]

    # Copy VLRs but update CRS
    # (In practice, we'd update the CRS VLR, but for now just copy)
    for vlr in las.header.vlrs:
        new_header.vlrs.append(vlr)

    new_las = laspy.LasData(new_header)
    new_las.x = x_new
    new_las.y = y_new
    new_las.z = z_new

    # Copy other attributes
    new_las.intensity = las.intensity
    new_las.classification = las.classification

    if hasattr(las, "red") and hasattr(las, "green") and hasattr(las, "blue"):
        new_las.red = las.red
        new_las.green = las.green
        new_las.blue = las.blue

    if hasattr(las, "return_number"):
        new_las.return_number = las.return_number
    if hasattr(las, "number_of_returns"):
        new_las.number_of_returns = las.number_of_returns

    offset = np.array(
        [
            np.mean(x_new) - np.mean(x_orig),
            np.mean(y_new) - np.mean(y_orig),
            0.0,
        ]
    )

    print("  Transformation complete")
    print(f"  New bounds X: [{x_new.min():.2f}, {x_new.max():.2f}]")
    print(f"  New bounds Y: [{y_new.min():.2f}, {y_new.max():.2f}]")

    return new_las, source_crs, offset


def extract_ground_points(
    las: laspy.LasData,
    classification: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract ground points from LAS data.

    Args:
        las: LAS data
        classification: Ground classification code (default: 2)

    Returns:
        Tuple of (xyz_points, mask) where mask indicates which points are ground
    """
    mask = np.array(las.classification) == classification
    points = np.vstack((las.x, las.y, las.z)).T

    ground_points = points[mask]
    print(
        f"  Ground points: {len(ground_points):,} / {len(points):,} ({100 * len(ground_points) / len(points):.1f}%)"
    )

    return ground_points, mask


def icp_align(
    source_points: np.ndarray,
    target_points: np.ndarray,
    max_correspondence_distance: float = 10.0,
    max_iterations: int = 50,
    fitness_threshold: float = 1e-6,
) -> tuple[np.ndarray, dict]:
    """
    Perform ICP alignment of source points to target points.

    Uses point-to-plane ICP, which is better suited for ground surfaces than
    point-to-point ICP. Point-to-plane minimizes the distance from source
    points to the tangent plane at corresponding target points, which handles
    the case where point densities differ between source and target.

    In the ground-to-all strategy:
    - source_points: ALL iPhone points
    - target_points: Reference GROUND points only

    Points in source that are farther than max_correspondence_distance from
    any target point won't contribute to the alignment. This naturally filters
    out non-ground iPhone points (trees, vegetation) since they won't have
    nearby correspondences in the ground-only target.

    Args:
        source_points: Nx3 array of source points (to be transformed).
            In ground-to-all strategy, this is ALL iPhone points.
        target_points: Mx3 array of target points (reference).
            In ground-to-all strategy, this is reference GROUND points only.
        max_correspondence_distance: Maximum distance for point correspondences.
            Points farther than this from any target point are ignored.
            Typical value: 10 file units (e.g., 10 feet for NYS State Plane).
        max_iterations: Maximum ICP iterations before stopping.
        fitness_threshold: Convergence threshold for relative fitness change.

    Returns:
        Tuple of (4x4 transformation matrix, alignment_info dict).
        alignment_info contains:
        - fitness: Fraction of source points with valid correspondences (0-1).
          May be low (<0.3) in ground-to-all strategy due to non-ground points.
        - inlier_rmse: RMS error of inlier correspondences.
        - translation_xyz: Translation component [tx, ty, tz].
        - rotation_angle_deg: Total rotation angle in degrees.
        - correspondence_count: Number of valid point correspondences.
    """
    print("  Running ICP alignment...")
    print(f"    Source points: {len(source_points):,}")
    print(f"    Target points: {len(target_points):,}")
    print(f"    Max correspondence distance: {max_correspondence_distance}")

    # Create Open3D point clouds
    source_pcd = o3d.geometry.PointCloud()
    source_pcd.points = o3d.utility.Vector3dVector(source_points)

    target_pcd = o3d.geometry.PointCloud()
    target_pcd.points = o3d.utility.Vector3dVector(target_points)

    # Initial identity transformation
    init_transform = np.eye(4)

    # First try point-to-point ICP (more robust for initial alignment)
    result = o3d.pipelines.registration.registration_icp(
        source_pcd,
        target_pcd,
        max_correspondence_distance,
        init_transform,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(
            max_iteration=max_iterations,
            relative_fitness=fitness_threshold,
            relative_rmse=fitness_threshold,
        ),
    )

    transform = result.transformation

    # Extract translation and rotation info for reporting
    translation = transform[:3, 3]
    rotation_matrix = transform[:3, :3]

    # Calculate rotation angle (approximate, assuming small rotations)
    rotation_angle_rad = np.arccos(np.clip((np.trace(rotation_matrix) - 1) / 2, -1, 1))
    rotation_angle_deg = np.degrees(rotation_angle_rad)

    info = {
        "fitness": result.fitness,
        "inlier_rmse": result.inlier_rmse,
        "translation_xyz": translation,
        "rotation_angle_deg": rotation_angle_deg,
        "correspondence_count": len(result.correspondence_set),
    }

    print("    ICP Results:")
    print(f"      Fitness: {result.fitness:.4f} (1.0 = perfect)")
    print(f"      Inlier RMSE: {result.inlier_rmse:.4f}")
    print(
        f"      Translation: [{translation[0]:.3f}, {translation[1]:.3f}, {translation[2]:.3f}]"
    )
    print(f"      Rotation: {rotation_angle_deg:.3f}°")
    print(f"      Correspondences: {len(result.correspondence_set):,}")

    return transform, info


def apply_transform_to_las(
    las: laspy.LasData,
    transform: np.ndarray,
) -> laspy.LasData:
    """
    Apply a 4x4 transformation matrix to LAS point coordinates.

    Args:
        las: Input LAS data
        transform: 4x4 transformation matrix

    Returns:
        New LAS data with transformed coordinates
    """
    points = np.vstack((las.x, las.y, las.z)).T

    # Apply transformation (homogeneous coordinates)
    ones = np.ones((len(points), 1))
    points_h = np.hstack((points, ones))
    transformed = (transform @ points_h.T).T[:, :3]

    # Create new LAS
    new_header = laspy.LasHeader(point_format=las.header.point_format, version="1.4")
    new_header.scales = las.header.scales
    new_header.offsets = [
        np.min(transformed[:, 0]),
        np.min(transformed[:, 1]),
        np.min(transformed[:, 2]),
    ]

    for vlr in las.header.vlrs:
        new_header.vlrs.append(vlr)

    new_las = laspy.LasData(new_header)
    new_las.x = transformed[:, 0]
    new_las.y = transformed[:, 1]
    new_las.z = transformed[:, 2]

    # Copy attributes
    new_las.intensity = las.intensity
    new_las.classification = las.classification

    if hasattr(las, "red") and hasattr(las, "green") and hasattr(las, "blue"):
        new_las.red = las.red
        new_las.green = las.green
        new_las.blue = las.blue

    if hasattr(las, "return_number"):
        new_las.return_number = las.return_number
    if hasattr(las, "number_of_returns"):
        new_las.number_of_returns = las.number_of_returns

    return new_las


def transfer_ground_classification(
    iphone_las: laspy.LasData,
    reference_ground_points: np.ndarray,
    horizontal_tolerance: float = 1.0,
    vertical_tolerance: float = 0.5,
) -> laspy.LasData:
    """
    Transfer ground classification from reference to iPhone points.

    iPhone scans from Polycam don't include ground classification. This function
    infers which iPhone points are ground by checking proximity to the reference
    ground surface (which has professional classification from NYS).

    Algorithm:
        For each iPhone point (x, y, z):
            1. Find all reference ground points within horizontal_tolerance of (x, y)
            2. For those nearby points, check if any have |z_ref - z_iphone| <= vertical_tolerance
            3. If yes, classify the iPhone point as ground (code 2)

    This two-stage approach (horizontal then vertical) handles terrain slopes:
    an iPhone point on a hillside should only be classified as ground if it's
    vertically close to the reference ground at that horizontal location.

    Why transfer classification?
        - Enables consistent filtering (e.g., extract ground-only for DEMs)
        - Allows downstream tools to distinguish ground from vegetation
        - Preserves professional classification work from NYS data

    Args:
        iphone_las: iPhone LAS data (will be modified). Should already be
            aligned to the reference CRS via ICP.
        reference_ground_points: Nx3 array of reference ground points.
            Extract these using extract_ground_points(ref_las, classification=2).
        horizontal_tolerance: Max horizontal (XY) distance to search for nearby
            reference ground points. In file units (typically feet for NYS).
            Default 1.0 = ~1 foot search radius.
        vertical_tolerance: Max vertical (Z) distance to consider a point as
            ground. In file units. Default 0.5 = ~6 inches vertical tolerance.

    Returns:
        New LAS data with classification field updated. Points classified as
        ground will have classification=2.
    """
    print("Transferring ground classification to iPhone points...")

    iphone_points = np.vstack((iphone_las.x, iphone_las.y, iphone_las.z)).T
    print(f"  iPhone points: {len(iphone_points):,}")
    print(f"  Reference ground points: {len(reference_ground_points):,}")
    print(f"  Horizontal tolerance: {horizontal_tolerance}")
    print(f"  Vertical tolerance: {vertical_tolerance}")

    # Build KD-tree of reference ground points (2D for horizontal search)
    ref_xy = reference_ground_points[:, :2]
    ref_z = reference_ground_points[:, 2]

    # Use Open3D KD-tree for efficient nearest neighbor search
    ref_pcd = o3d.geometry.PointCloud()
    ref_pcd.points = o3d.utility.Vector3dVector(
        np.column_stack([ref_xy, np.zeros(len(ref_xy))])  # 2D search
    )
    ref_tree = o3d.geometry.KDTreeFlann(ref_pcd)

    # For each iPhone point, find nearby reference ground points
    iphone_xy = iphone_points[:, :2]
    iphone_z = iphone_points[:, 2]

    ground_mask = np.zeros(len(iphone_points), dtype=bool)

    # Batch process for efficiency
    for i in range(len(iphone_points)):
        query_point = np.array([iphone_xy[i, 0], iphone_xy[i, 1], 0.0])

        # Find points within horizontal tolerance
        [k, idx, dist_sq] = ref_tree.search_radius_vector_3d(
            query_point, horizontal_tolerance
        )

        if k > 0:
            # Check vertical distance to nearest ground points
            nearby_z = ref_z[idx]
            z_diff = np.abs(iphone_z[i] - nearby_z)

            if np.min(z_diff) <= vertical_tolerance:
                ground_mask[i] = True

    # Update classification
    new_classification = np.array(iphone_las.classification)
    new_classification[ground_mask] = 2  # Ground classification

    ground_count = np.sum(ground_mask)
    print(
        f"  Points classified as ground: {ground_count:,} ({100 * ground_count / len(iphone_points):.1f}%)"
    )

    # Create new LAS with updated classification
    new_header = laspy.LasHeader(
        point_format=iphone_las.header.point_format, version="1.4"
    )
    new_header.scales = iphone_las.header.scales
    new_header.offsets = iphone_las.header.offsets

    for vlr in iphone_las.header.vlrs:
        new_header.vlrs.append(vlr)

    new_las = laspy.LasData(new_header)
    new_las.x = iphone_las.x
    new_las.y = iphone_las.y
    new_las.z = iphone_las.z
    new_las.intensity = iphone_las.intensity
    new_las.classification = new_classification

    if hasattr(iphone_las, "red"):
        new_las.red = iphone_las.red
        new_las.green = iphone_las.green
        new_las.blue = iphone_las.blue

    if hasattr(iphone_las, "return_number"):
        new_las.return_number = iphone_las.return_number
    if hasattr(iphone_las, "number_of_returns"):
        new_las.number_of_returns = iphone_las.number_of_returns

    return new_las


def merge_with_replacement(
    reference_las: laspy.LasData,
    iphone_las: laspy.LasData,
    no_replace: bool = False,
) -> laspy.LasData:
    """
    Merge two point clouds, optionally replacing reference points in overlap region.

    Strategy: REPLACEMENT (default) or BLEND
    -----------------------------------------
    By default, iPhone data completely replaces NYS data in the overlap region.
    If no_replace=True, all points from both clouds are kept (blended).

    Replacement approach:
    1. Avoids double-density artifacts at boundaries
    2. Provides clear data provenance (each point comes from one source)
    3. Prioritizes higher-resolution iPhone data where available
    4. Simplifies downstream processing

    The overlap region is defined by the convex hull of the iPhone scan points.

    Visualization (replacement mode):
        ┌─────────────────────────────────────┐
        │                                     │
        │         NYS Data (kept)             │
        │                                     │
        │      ┌───────────────────┐          │
        │      │                   │          │
        │      │   iPhone Data     │          │
        │      │   (replaces NYS)  │          │
        │      │                   │          │
        │      └───────────────────┘          │
        │                                     │
        │         NYS Data (kept)             │
        │                                     │
        └─────────────────────────────────────┘

    Args:
        reference_las: Reference LAS data (NYS). Lower resolution but covers
            larger area. Points within the iPhone scan convex hull will be
            excluded unless no_replace=True.
        iphone_las: iPhone LAS data (from Polycam). Higher resolution, takes
            priority in the overlap region. Should be aligned via ICP first.
        no_replace: If True, keep all reference points (blend instead of replace).

    Returns:
        Merged LAS data with:
        - Reference points (all if no_replace, else only outside convex hull)
        - All iPhone points
        - Attributes (RGB, classification, intensity) preserved from both sources
    """
    if no_replace:
        print("Merging point clouds (blending - keeping all points)...")
    else:
        print("Merging point clouds with replacement...")

    ref_points = np.vstack((reference_las.x, reference_las.y, reference_las.z)).T
    iphone_points = np.vstack((iphone_las.x, iphone_las.y, iphone_las.z)).T

    print(f"  Reference points: {len(ref_points):,}")
    print(f"  iPhone points: {len(iphone_points):,}")

    # Compute iPhone scan center for metadata
    iphone_center = np.mean(iphone_points[:, :2], axis=0)
    print(f"  iPhone scan center: ({iphone_center[0]:.2f}, {iphone_center[1]:.2f})")

    # Compute convex hull of iPhone points for boundary
    hull = ConvexHull(iphone_points[:, :2])
    boundary_polygon = iphone_points[hull.vertices, :2]
    print(f"  iPhone boundary: convex hull with {len(boundary_polygon)} vertices")

    if no_replace:
        # Keep all reference points
        keep_mask = np.ones(len(ref_points), dtype=bool)
        print(f"  Blending mode: keeping all {len(ref_points):,} reference points")
    else:
        # Find reference points outside the iPhone scan convex hull
        # Use matplotlib's Path for point-in-polygon test
        from matplotlib.path import Path

        hull_path = Path(boundary_polygon)
        inside_hull = hull_path.contains_points(ref_points[:, :2])
        keep_mask = ~inside_hull

        print(f"  Reference points kept (outside hull): {np.sum(keep_mask):,}")
        print(f"  Reference points replaced: {np.sum(~keep_mask):,}")

    kept_ref_points = ref_points[keep_mask]

    # Build output LAS
    # Use reference header as template (preserves CRS)
    total_points = len(kept_ref_points) + len(iphone_points)
    print(f"  Total merged points: {total_points:,}")

    # Combine coordinates
    merged_x = np.concatenate([kept_ref_points[:, 0], iphone_points[:, 0]])
    merged_y = np.concatenate([kept_ref_points[:, 1], iphone_points[:, 1]])
    merged_z = np.concatenate([kept_ref_points[:, 2], iphone_points[:, 2]])

    # Check if either source has RGB to determine output format
    def has_rgb(las):
        try:
            _ = las.red
            return True
        except Exception:
            return False

    either_has_rgb = has_rgb(reference_las) or has_rgb(iphone_las)

    # Determine output point format - ensure it supports RGB if either input has it
    source_format = reference_las.header.point_format.id
    if either_has_rgb:
        # Map non-RGB formats to their RGB equivalents
        format_mapping = {
            0: 2,  # Format 0 -> Format 2 (adds RGB)
            1: 3,  # Format 1 -> Format 3 (adds RGB)
            6: 7,  # Format 6 -> Format 7 (adds RGB)
            9: 10, # Format 9 -> Format 10 (adds RGB)
        }
        output_format = format_mapping.get(source_format, source_format)
        # If source already supports RGB, keep it
        if source_format in [2, 3, 5, 7, 8, 10]:
            output_format = source_format
    else:
        output_format = source_format

    # Create new header
    new_header = laspy.LasHeader(
        point_format=output_format, version="1.4"
    )
    new_header.scales = reference_las.header.scales
    new_header.offsets = [np.min(merged_x), np.min(merged_y), np.min(merged_z)]

    for vlr in reference_las.header.vlrs:
        new_header.vlrs.append(vlr)

    # Add iPhone scan boundary as custom VLR (reuse convex hull computed earlier)
    boundary_data = {
        "type": "iphone_scan_boundary",
        "polygon": boundary_polygon.tolist(),
        "center": iphone_center.tolist(),
    }
    boundary_vlr = laspy.VLR(
        user_id="self-survey",
        record_id=1001,
        record_data=json.dumps(boundary_data).encode("utf-8"),
        description="iPhone scan boundary from registration",
    )
    new_header.vlrs.append(boundary_vlr)

    merged_las = laspy.LasData(new_header)
    merged_las.x = merged_x
    merged_las.y = merged_y
    merged_las.z = merged_z

    # Merge attributes
    ref_intensity = np.array(reference_las.intensity)[keep_mask]
    iphone_intensity = np.array(iphone_las.intensity)
    merged_las.intensity = np.concatenate([ref_intensity, iphone_intensity])

    ref_classification = np.array(reference_las.classification)[keep_mask]
    iphone_classification = np.array(iphone_las.classification)
    merged_las.classification = np.concatenate(
        [ref_classification, iphone_classification]
    )

    # Check RGB support (reuse has_rgb from above)
    ref_has_rgb = has_rgb(reference_las)
    iphone_has_rgb = has_rgb(iphone_las)

    print(f"  Reference has RGB: {ref_has_rgb}")
    print(f"  iPhone has RGB: {iphone_has_rgb}")

    # Helper to scale 8-bit RGB to 16-bit if needed
    def scale_rgb_to_16bit(rgb_array):
        rgb = np.array(rgb_array)
        # If max value is <= 255, it's 8-bit and needs scaling
        if rgb.max() <= 255:
            return (rgb.astype(np.uint16) * 257).astype(np.uint16)  # 257 = 65535/255
        return rgb.astype(np.uint16)

    if ref_has_rgb and iphone_has_rgb:
        ref_red = scale_rgb_to_16bit(reference_las.red)[keep_mask]
        ref_green = scale_rgb_to_16bit(reference_las.green)[keep_mask]
        ref_blue = scale_rgb_to_16bit(reference_las.blue)[keep_mask]
        iphone_red = scale_rgb_to_16bit(iphone_las.red)
        iphone_green = scale_rgb_to_16bit(iphone_las.green)
        iphone_blue = scale_rgb_to_16bit(iphone_las.blue)

        merged_las.red = np.concatenate([ref_red, iphone_red])
        merged_las.green = np.concatenate([ref_green, iphone_green])
        merged_las.blue = np.concatenate([ref_blue, iphone_blue])
    elif ref_has_rgb:
        # Reference has RGB, iPhone doesn't - fill iPhone with white
        merged_las.red = np.concatenate(
            [
                scale_rgb_to_16bit(reference_las.red)[keep_mask],
                np.full(len(iphone_points), 65535, dtype=np.uint16),
            ]
        )
        merged_las.green = np.concatenate(
            [
                scale_rgb_to_16bit(reference_las.green)[keep_mask],
                np.full(len(iphone_points), 65535, dtype=np.uint16),
            ]
        )
        merged_las.blue = np.concatenate(
            [
                scale_rgb_to_16bit(reference_las.blue)[keep_mask],
                np.full(len(iphone_points), 65535, dtype=np.uint16),
            ]
        )
    elif iphone_has_rgb:
        # iPhone has RGB, reference doesn't - fill reference with white
        merged_las.red = np.concatenate(
            [
                np.full(np.sum(keep_mask), 65535, dtype=np.uint16),
                scale_rgb_to_16bit(iphone_las.red),
            ]
        )
        merged_las.green = np.concatenate(
            [
                np.full(np.sum(keep_mask), 65535, dtype=np.uint16),
                scale_rgb_to_16bit(iphone_las.green),
            ]
        )
        merged_las.blue = np.concatenate(
            [
                np.full(np.sum(keep_mask), 65535, dtype=np.uint16),
                scale_rgb_to_16bit(iphone_las.blue),
            ]
        )

    return merged_las


def get_iphone_boundary_from_las(las: laspy.LasData) -> dict | None:
    """
    Read iPhone scan boundary VLR if present in a LAS file.

    The boundary is stored during merge_with_replacement() as a custom VLR
    with user_id="self-survey" and record_id=1001.

    Args:
        las: LAS data to check for boundary VLR

    Returns:
        Dictionary with boundary data if found:
        - "type": "iphone_scan_boundary"
        - "polygon": List of [x, y] coordinates forming the convex hull
        - "center": [x, y] center point of the iPhone scan
        Returns None if no boundary VLR is present.
    """
    for vlr in las.header.vlrs:
        if vlr.user_id == "self-survey" and vlr.record_id == 1001:
            return json.loads(vlr.record_data.decode("utf-8"))
    return None
