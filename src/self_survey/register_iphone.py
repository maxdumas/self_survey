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

import laspy
import numpy as np
import open3d as o3d
from pyproj import CRS, Transformer

from self_survey.clip_to_radius import get_crs_from_las

__all__ = [
    "load_and_transform_to_crs",
    "extract_ground_points",
    "icp_align",
    "apply_transform_to_las",
    "transfer_ground_classification",
    "merge_with_replacement",
]


def load_and_transform_to_crs(
    las_path: str,
    target_crs: CRS,
) -> tuple[laspy.LasData, CRS, np.ndarray | None]:
    """
    Load a LAS file and transform coordinates to target CRS if needed.

    Args:
        las_path: Path to LAS/LAZ file
        target_crs: Target coordinate reference system

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
        print("  Assuming coordinates are already in target CRS")
        return las, target_crs, None

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

    # Estimate normals for point-to-plane ICP (better for ground surfaces)
    source_pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=5.0, max_nn=30)
    )
    target_pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=5.0, max_nn=30)
    )

    # Initial identity transformation
    init_transform = np.eye(4)

    # Run point-to-plane ICP
    result = o3d.pipelines.registration.registration_icp(
        source_pcd,
        target_pcd,
        max_correspondence_distance,
        init_transform,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
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
    replacement_radius: float | None = None,
) -> laspy.LasData:
    """
    Merge two point clouds, replacing reference points in overlap region with iPhone points.

    Strategy: REPLACEMENT, not blending
    ------------------------------------
    In the overlap region, iPhone data completely replaces NYS data rather than
    being blended or merged. This approach:

    1. Avoids double-density artifacts at boundaries
    2. Provides clear data provenance (each point comes from one source)
    3. Prioritizes higher-resolution iPhone data where available
    4. Simplifies downstream processing

    The overlap region is defined as a circle centered on the iPhone scan's
    centroid with radius equal to the scan's maximum extent (plus 10% buffer).

    Visualization:
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
            larger area. Points within the replacement region will be excluded.
        iphone_las: iPhone LAS data (from Polycam). Higher resolution, takes
            priority in the overlap region. Should be aligned via ICP first.
        replacement_radius: If provided, defines the circular region around the
            iPhone scan center where reference points are excluded. If None,
            auto-calculated as 1.1 × (max distance from centroid to any iPhone point).

    Returns:
        Merged LAS data with:
        - Reference points outside the replacement region
        - All iPhone points
        - Attributes (RGB, classification, intensity) preserved from both sources
    """
    print("Merging point clouds with replacement...")

    ref_points = np.vstack((reference_las.x, reference_las.y, reference_las.z)).T
    iphone_points = np.vstack((iphone_las.x, iphone_las.y, iphone_las.z)).T

    print(f"  Reference points: {len(ref_points):,}")
    print(f"  iPhone points: {len(iphone_points):,}")

    # Determine replacement region from iPhone scan extent
    iphone_center = np.mean(iphone_points[:, :2], axis=0)
    iphone_extent = np.max(np.linalg.norm(iphone_points[:, :2] - iphone_center, axis=1))

    if replacement_radius is None:
        # Use iPhone scan extent as replacement radius
        replacement_radius = iphone_extent * 1.1  # 10% buffer

    print(f"  iPhone scan center: ({iphone_center[0]:.2f}, {iphone_center[1]:.2f})")
    print(f"  iPhone scan extent: {iphone_extent:.2f}")
    print(f"  Replacement radius: {replacement_radius:.2f}")

    # Find reference points outside the iPhone scan region
    # Using a simple circular region around iPhone center
    ref_distances = np.linalg.norm(ref_points[:, :2] - iphone_center, axis=1)
    keep_mask = ref_distances > replacement_radius

    kept_ref_points = ref_points[keep_mask]
    print(f"  Reference points kept (outside overlap): {len(kept_ref_points):,}")
    print(f"  Reference points replaced: {np.sum(~keep_mask):,}")

    # Build output LAS
    # Use reference header as template (preserves CRS)
    total_points = len(kept_ref_points) + len(iphone_points)
    print(f"  Total merged points: {total_points:,}")

    # Combine coordinates
    merged_x = np.concatenate([kept_ref_points[:, 0], iphone_points[:, 0]])
    merged_y = np.concatenate([kept_ref_points[:, 1], iphone_points[:, 1]])
    merged_z = np.concatenate([kept_ref_points[:, 2], iphone_points[:, 2]])

    # Create new header
    new_header = laspy.LasHeader(
        point_format=reference_las.header.point_format, version="1.4"
    )
    new_header.scales = reference_las.header.scales
    new_header.offsets = [np.min(merged_x), np.min(merged_y), np.min(merged_z)]

    for vlr in reference_las.header.vlrs:
        new_header.vlrs.append(vlr)

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

    # Merge RGB if both have it
    ref_has_rgb = hasattr(reference_las, "red")
    iphone_has_rgb = hasattr(iphone_las, "red")

    if ref_has_rgb and iphone_has_rgb:
        merged_las.red = np.concatenate(
            [np.array(reference_las.red)[keep_mask], np.array(iphone_las.red)]
        )
        merged_las.green = np.concatenate(
            [np.array(reference_las.green)[keep_mask], np.array(iphone_las.green)]
        )
        merged_las.blue = np.concatenate(
            [np.array(reference_las.blue)[keep_mask], np.array(iphone_las.blue)]
        )
    elif ref_has_rgb:
        # Reference has RGB, iPhone doesn't - fill with white
        merged_las.red = np.concatenate(
            [
                np.array(reference_las.red)[keep_mask],
                np.full(len(iphone_points), 65535, dtype=np.uint16),
            ]
        )
        merged_las.green = np.concatenate(
            [
                np.array(reference_las.green)[keep_mask],
                np.full(len(iphone_points), 65535, dtype=np.uint16),
            ]
        )
        merged_las.blue = np.concatenate(
            [
                np.array(reference_las.blue)[keep_mask],
                np.full(len(iphone_points), 65535, dtype=np.uint16),
            ]
        )

    return merged_las
