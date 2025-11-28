"""
Register command for iPhone LiDAR scan alignment and merging.

This command handles registering an iPhone LiDAR scan (from Polycam or similar)
to NYS reference data and producing a merged point cloud.

Pipeline Steps
--------------

1. **Load**: Read both point clouds, detect CRS from metadata
2. **Transform**: Convert iPhone coordinates to reference CRS (e.g., WGS84 → State Plane)
3. **ICP Alignment**: Fine-tune alignment using ground-to-all ICP strategy
4. **Classification Transfer**: Infer ground points in iPhone data from NYS ground surface
5. **Merge**: Combine clouds with iPhone replacing NYS in overlap region
6. **Save**: Write merged LAZ file

Ground-to-All ICP Strategy
--------------------------

Since Polycam doesn't classify ground points, we can't do ground-to-ground ICP.
Instead:

- Target: NYS GROUND points only (classification=2)
- Source: ALL iPhone points

ICP's max_correspondence_distance naturally filters non-ground iPhone points
(trees, vegetation) since they won't have nearby correspondences in the
ground-only target. This aligns the ground surfaces, which is the most
reliable common feature between aerial and iPhone LiDAR.

Low fitness scores (0.1-0.3) are expected because many iPhone points
(non-ground) won't have correspondences. Check translation values to
verify alignment is reasonable.

See Also
--------
register_iphone module for detailed algorithmic documentation.
"""

from pathlib import Path
from typing import Annotated

import cyclopts
import laspy
import numpy as np


def register(
    iphone_scan: Annotated[
        Path,
        cyclopts.Parameter(
            help="iPhone LiDAR scan (LAS/LAZ from Polycam or similar)",
        ),
    ],
    *,
    reference: Annotated[
        Path,
        cyclopts.Parameter(
            name="--reference",
            alias="-r",
            help="Reference LAS/LAZ file (NYS LiDAR, output from 'ingest' command)",
        ),
    ],
    output: Annotated[
        Path,
        cyclopts.Parameter(
            name="--output",
            alias="-o",
            help="Output merged LAZ/LAS file path",
        ),
    ],
    lat: Annotated[
        float,
        cyclopts.Parameter(
            name="--lat",
            help="Approximate latitude (WGS84) where the iPhone scan was taken. "
            "Required for scans in local coordinates (no CRS metadata).",
        ),
    ],
    lon: Annotated[
        float,
        cyclopts.Parameter(
            name="--lon",
            help="Approximate longitude (WGS84) where the iPhone scan was taken. "
            "Required for scans in local coordinates (no CRS metadata).",
        ),
    ],
    icp_distance: Annotated[
        float,
        cyclopts.Parameter(
            help="Maximum correspondence distance for ICP alignment (in file units)",
        ),
    ] = 10.0,
    icp_iterations: Annotated[
        int,
        cyclopts.Parameter(
            help="Maximum ICP iterations",
        ),
    ] = 50,
    transfer_classification: Annotated[
        bool,
        cyclopts.Parameter(
            help="Transfer ground classification from reference to iPhone points",
        ),
    ] = True,
    horizontal_tolerance: Annotated[
        float,
        cyclopts.Parameter(
            help="Horizontal tolerance for ground classification transfer (file units)",
        ),
    ] = 1.0,
    vertical_tolerance: Annotated[
        float,
        cyclopts.Parameter(
            help="Vertical tolerance for ground classification transfer (file units)",
        ),
    ] = 0.5,
    replacement_radius: Annotated[
        float | None,
        cyclopts.Parameter(
            help="Radius around iPhone scan center to replace reference points. "
            "If not specified, auto-calculated from iPhone scan extent.",
        ),
    ] = None,
    skip_icp: Annotated[
        bool,
        cyclopts.Parameter(
            help="Skip ICP alignment (use only if scans are already well-aligned)",
        ),
    ] = False,
    no_replace: Annotated[
        bool,
        cyclopts.Parameter(
            help="Don't replace reference points - blend both point clouds together",
        ),
    ] = False,
) -> None:
    """
    Register an iPhone LiDAR scan to NYS reference data and merge.

    This command aligns a georeferenced iPhone LiDAR scan (from Polycam or similar)
    to NYS reference data and produces a merged point cloud where iPhone data
    replaces NYS data in the overlap region.
    """
    from self_survey.clip_to_radius import get_crs_from_las, transform_latlon_to_crs
    from self_survey.register_iphone import (
        apply_transform_to_las,
        extract_ground_points,
        icp_align,
        load_and_transform_to_crs,
        merge_with_replacement,
        transfer_ground_classification,
    )

    # Validate inputs
    if not iphone_scan.exists():
        raise cyclopts.ValidationError(f"iPhone scan not found: {iphone_scan}")

    if not reference.exists():
        raise cyclopts.ValidationError(f"Reference file not found: {reference}")

    # Step 1: Load reference data
    print("=" * 60)
    print("STEP 1: Loading reference data")
    print("=" * 60)

    print(f"\nLoading {reference}...")
    ref_las = laspy.read(str(reference))
    ref_crs = get_crs_from_las(ref_las)

    if ref_crs is None:
        raise cyclopts.ValidationError(
            "Could not detect CRS from reference file. "
            "Please ensure the reference file has CRS metadata."
        )

    print(f"  Reference CRS: {ref_crs.name}")
    print(f"  Reference points: {len(ref_las.points):,}")
    print(f"  Bounds X: [{ref_las.x.min():.2f}, {ref_las.x.max():.2f}]")
    print(f"  Bounds Y: [{ref_las.y.min():.2f}, {ref_las.y.max():.2f}]")
    print(f"  Bounds Z: [{ref_las.z.min():.2f}, {ref_las.z.max():.2f}]")

    # Step 2: Load and transform iPhone scan
    print("\n" + "=" * 60)
    print("STEP 2: Loading and transforming iPhone scan")
    print("=" * 60)

    # Transform the provided lat/lon to the reference CRS
    # This will be used as the center point for local-coordinate scans
    initial_x, initial_y = transform_latlon_to_crs(lat, lon, ref_crs)
    print(f"\nInitial position (from --lat/--lon):")
    print(f"  WGS84: ({lat}, {lon})")
    print(f"  Reference CRS: ({initial_x:.2f}, {initial_y:.2f})")

    iphone_las, iphone_crs, transform_offset = load_and_transform_to_crs(
        str(iphone_scan), ref_crs,
        initial_position=(initial_x, initial_y),
        reference_las=ref_las,
    )

    print(f"  iPhone points: {len(iphone_las.points):,}")
    print(f"  Bounds X: [{iphone_las.x.min():.2f}, {iphone_las.x.max():.2f}]")
    print(f"  Bounds Y: [{iphone_las.y.min():.2f}, {iphone_las.y.max():.2f}]")
    print(f"  Bounds Z: [{iphone_las.z.min():.2f}, {iphone_las.z.max():.2f}]")

    # Step 3: ICP alignment
    # Strategy: Use reference GROUND points as target, ALL iPhone points as source
    # The ICP correspondence distance will naturally filter out non-ground iPhone points
    if not skip_icp:
        print("\n" + "=" * 60)
        print("STEP 3: ICP alignment (ground-to-all strategy)")
        print("=" * 60)

        # Extract ground points from reference (target)
        print("\nExtracting ground points from reference...")
        ref_ground, ref_ground_mask = extract_ground_points(ref_las, classification=2)

        if len(ref_ground) < 100:
            print(f"\n  WARNING: Only {len(ref_ground)} ground points in reference.")
            print("  Consider checking that reference data has ground classification.")

        # Use ALL iPhone points as source
        print("\nUsing all iPhone points for alignment...")
        iphone_all_points = np.vstack((iphone_las.x, iphone_las.y, iphone_las.z)).T
        print(f"  iPhone points: {len(iphone_all_points):,}")

        # Filter reference ground points to those near the iPhone scan
        # This improves ICP performance and avoids confusion from distant points
        iphone_x_min, iphone_x_max = iphone_las.x.min(), iphone_las.x.max()
        iphone_y_min, iphone_y_max = iphone_las.y.min(), iphone_las.y.max()
        iphone_z_min, iphone_z_max = iphone_las.z.min(), iphone_las.z.max()

        # Add buffer around iPhone bounds for ICP search
        buffer = icp_distance * 2
        ref_ground_in_xy = (
            (ref_ground[:, 0] >= iphone_x_min - buffer) & (ref_ground[:, 0] <= iphone_x_max + buffer) &
            (ref_ground[:, 1] >= iphone_y_min - buffer) & (ref_ground[:, 1] <= iphone_y_max + buffer)
        )
        ref_ground_nearby = ref_ground[ref_ground_in_xy]
        print(f"\n  Reference ground points near iPhone scan: {len(ref_ground_nearby):,}")

        if len(ref_ground_nearby) > 0:
            print(f"    Z range of those points: [{ref_ground_nearby[:, 2].min():.2f}, {ref_ground_nearby[:, 2].max():.2f}]")
            print(f"    iPhone Z range: [{iphone_z_min:.2f}, {iphone_z_max:.2f}]")
        else:
            print("  ERROR: No reference ground points found near iPhone scan!")
            print("  Check that --lat/--lon are correct.")

        # Run ICP: iPhone all points -> nearby Reference ground points
        # Non-ground iPhone points won't find correspondences within icp_distance
        print("\nRunning ICP alignment...")
        print("  (iPhone all points → Reference ground points)")
        transform, icp_info = icp_align(
            source_points=iphone_all_points,
            target_points=ref_ground_nearby,
            max_correspondence_distance=icp_distance,
            max_iterations=icp_iterations,
        )

        # Check alignment quality
        if icp_info["correspondence_count"] == 0:
            print("\n  ERROR: ICP found no correspondences!")
            print("  The point clouds may not be overlapping.")
            print("  Skipping ICP transformation to preserve initial alignment.")
            print("  Check that --lat/--lon are correct and --icp-distance is large enough.")
        elif icp_info["fitness"] < 0.1:
            print("\n  WARNING: Low ICP fitness score.")
            print("  This is expected if the iPhone scan has many non-ground points.")
            print(
                "  Check the translation values to verify alignment looks reasonable."
            )
            # Apply transformation to iPhone scan
            print("\nApplying transformation to iPhone scan...")
            iphone_las = apply_transform_to_las(iphone_las, transform)
        else:
            # Apply transformation to iPhone scan
            print("\nApplying transformation to iPhone scan...")
            iphone_las = apply_transform_to_las(iphone_las, transform)

        print(f"  New bounds X: [{iphone_las.x.min():.2f}, {iphone_las.x.max():.2f}]")
        print(f"  New bounds Y: [{iphone_las.y.min():.2f}, {iphone_las.y.max():.2f}]")
        print(f"  New bounds Z: [{iphone_las.z.min():.2f}, {iphone_las.z.max():.2f}]")
    else:
        print("\n" + "=" * 60)
        print("STEP 3: Skipping ICP alignment (--skip-icp)")
        print("=" * 60)
        ref_ground = None

    # Step 4: Transfer ground classification
    if transfer_classification:
        print("\n" + "=" * 60)
        print("STEP 4: Transferring ground classification")
        print("=" * 60)

        # Get reference ground points if we didn't already extract them
        if ref_ground is None:
            print("\nExtracting ground points from reference...")
            ref_ground, _ = extract_ground_points(ref_las, classification=2)

        iphone_las = transfer_ground_classification(
            iphone_las,
            ref_ground,
            horizontal_tolerance=horizontal_tolerance,
            vertical_tolerance=vertical_tolerance,
        )
    else:
        print("\n" + "=" * 60)
        print("STEP 4: Skipping classification transfer (--no-transfer-classification)")
        print("=" * 60)

    # Step 5: Merge with replacement
    print("\n" + "=" * 60)
    print("STEP 5: Merging point clouds")
    print("=" * 60)

    merged_las = merge_with_replacement(
        reference_las=ref_las,
        iphone_las=iphone_las,
        replacement_radius=replacement_radius,
        no_replace=no_replace,
    )

    # Step 6: Save output
    print("\n" + "=" * 60)
    print("STEP 6: Saving output")
    print("=" * 60)

    print(f"\nSaving to {output}...")
    merged_las.write(str(output))
    file_size = output.stat().st_size / (1024 * 1024)
    print(f"  File size: {file_size:.1f} MB")

    print("\n" + "=" * 60)
    print("DONE!")
    print("=" * 60)
    print(f"\nOutput: {output}")
    print(f"Total points: {len(merged_las.points):,}")
