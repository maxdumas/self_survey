"""
Preprocess LiDAR data for property surveys.

Commands:
    ingest   - Merge NYS LiDAR tiles, drape orthoimagery, and clip to radius
    register - Register iPhone LiDAR scan to NYS reference and merge

Usage:
    preprocess ingest tile1.laz tile2.laz --ortho ortho.tif \
        --lat 42.4532 --lon -73.7891 --radius 100 -o output.laz

    preprocess register iphone_scan.laz --reference nys_data.laz -o merged.laz
"""

from pathlib import Path
from typing import Annotated

import cyclopts
import laspy
import numpy as np

app = cyclopts.App(
    name="preprocess",
    help="Preprocess LiDAR data for property surveys.",
)


@app.command(name="ingest")
def ingest(
    tiles: Annotated[
        list[Path],
        cyclopts.Parameter(
            help="One or more LAS/LAZ files to merge and process",
        ),
    ],
    *,
    lat: Annotated[
        float,
        cyclopts.Parameter(help="Center latitude (WGS84) for clipping"),
    ],
    lon: Annotated[
        float,
        cyclopts.Parameter(help="Center longitude (WGS84) for clipping"),
    ],
    radius: Annotated[
        float,
        cyclopts.Parameter(help="Clip radius"),
    ],
    output: Annotated[
        Path,
        cyclopts.Parameter(
            "--output",
            "-o",
            help="Output LAZ/LAS file path",
        ),
    ],
    ortho: Annotated[
        list[Path] | None,
        cyclopts.Parameter(
            help="Orthoimagery TIFF file(s) to drape over the point cloud. "
            "If multiple orthos are provided, they are applied in order.",
        ),
    ] = None,
    radius_units: Annotated[
        str,
        cyclopts.Parameter(
            help="Units for radius: 'meters', 'feet', or 'same' (as file units)",
        ),
    ] = "meters",
    epsg: Annotated[
        int | None,
        cyclopts.Parameter(
            help="EPSG code if auto-detection fails (e.g., 2261 for NY State Plane East)",
        ),
    ] = None,
    filter_ground: Annotated[
        bool,
        cyclopts.Parameter(
            help="Keep only ground points (classification=2)",
        ),
    ] = False,
) -> None:
    """
    Ingest NYS LiDAR data: merge tiles, colorize from orthoimagery, and clip to radius.

    The processing pipeline:
    1. Load and merge all input LAS/LAZ tiles
    2. If orthoimagery is provided, drape RGB colors onto the point cloud
    3. Clip to a circular region around the specified lat/lon point
    4. Save the result
    """
    import open3d as o3d
    from pyproj import CRS

    from self_survey.merge_lidar_tiles import (
        load_las_to_open3d,
        merge_point_clouds,
        save_as_laz,
    )
    from self_survey.colorize_from_ortho import colorize_point_cloud
    from self_survey.clip_to_radius import (
        get_crs_from_las,
        transform_latlon_to_crs,
        clip_to_radius,
    )

    # Validate inputs
    for tile in tiles:
        if not tile.exists():
            raise cyclopts.ValidationError(f"Input tile not found: {tile}")

    if ortho:
        for ortho_file in ortho:
            if not ortho_file.exists():
                raise cyclopts.ValidationError(f"Ortho file not found: {ortho_file}")

    if radius_units not in ("meters", "feet", "same"):
        raise cyclopts.ValidationError(
            f"Invalid radius_units: {radius_units}. Must be 'meters', 'feet', or 'same'"
        )

    # Step 1: Load and merge tiles
    print("=" * 60)
    print("STEP 1: Loading and merging LiDAR tiles")
    print("=" * 60)

    merged_pcd = None
    merged_meta = None

    for i, tile in enumerate(tiles):
        pcd, meta = load_las_to_open3d(str(tile))

        if merged_pcd is None:
            merged_pcd = pcd
            merged_meta = meta
        else:
            merged_pcd, merged_meta = merge_point_clouds(
                merged_pcd, merged_meta, pcd, meta
            )

    assert merged_pcd is not None
    assert merged_meta is not None

    print(f"\nMerged {len(tiles)} tile(s): {len(merged_pcd.points):,} total points")

    # Optional: filter to ground only
    if filter_ground:
        print("\nFiltering to ground points only (classification=2)...")
        mask = merged_meta["classification"] == 2

        points = np.asarray(merged_pcd.points)[mask]
        merged_pcd.points = o3d.utility.Vector3dVector(points)

        if merged_pcd.has_colors():
            colors = np.asarray(merged_pcd.colors)[mask]
            merged_pcd.colors = o3d.utility.Vector3dVector(colors)

        merged_meta["classification"] = merged_meta["classification"][mask]
        merged_meta["intensity"] = merged_meta["intensity"][mask]
        if merged_meta.get("return_number") is not None:
            merged_meta["return_number"] = merged_meta["return_number"][mask]
            merged_meta["number_of_returns"] = merged_meta["number_of_returns"][mask]

        print(f"  Ground points: {len(points):,}")

    # Step 2: Colorize from orthoimagery (if provided)
    if ortho:
        print("\n" + "=" * 60)
        print("STEP 2: Colorizing from orthoimagery")
        print("=" * 60)

        for ortho_file in ortho:
            print(f"\nProcessing {ortho_file.name}...")
            merged_pcd = colorize_point_cloud(merged_pcd, str(ortho_file))

    # Step 3: Clip to radius
    print("\n" + "=" * 60)
    print("STEP 3: Clipping to radius around center point")
    print("=" * 60)

    # We need to work with the raw laspy data for CRS detection and clipping
    # Save merged to temp file, then reload for clipping
    # This is a workaround since Open3D doesn't preserve CRS info
    temp_output = output.with_suffix(".temp.laz")
    save_as_laz(merged_pcd, merged_meta, str(temp_output))

    # Reload for CRS-aware clipping
    las = laspy.read(str(temp_output))

    # Determine CRS
    if epsg:
        crs = CRS.from_epsg(epsg)
        print(f"  Using specified EPSG:{epsg}")
    else:
        crs = get_crs_from_las(las)
        if crs:
            print(f"  Detected CRS: {crs.name}")
        else:
            temp_output.unlink()  # Clean up temp file
            raise cyclopts.ValidationError(
                "Could not detect CRS from file. Please specify --epsg manually.\n"
                "Common NY State Plane codes:\n"
                "  EPSG:2261 - NY State Plane East (NAD83, feet)\n"
                "  EPSG:2262 - NY State Plane Central (NAD83, feet)\n"
                "  EPSG:2263 - NY State Plane West (NAD83, feet)\n"
                "  EPSG:32618 - UTM Zone 18N (NAD83, meters)"
            )

    # Determine file units
    file_units = "unknown"
    if crs.axis_info:
        unit_name = crs.axis_info[0].unit_name.lower()
        if "foot" in unit_name or "feet" in unit_name or "ft" in unit_name:
            file_units = "feet"
        elif "metre" in unit_name or "meter" in unit_name:
            file_units = "meters"
    print(f"  File units: {file_units}")

    # Transform center point to file CRS
    center_x, center_y = transform_latlon_to_crs(lat, lon, crs)
    print(f"\nCenter point:")
    print(f"  WGS84: ({lat}, {lon})")
    print(f"  File CRS: ({center_x:.2f}, {center_y:.2f})")

    # Convert radius to file units
    clip_radius = radius
    if radius_units == "meters" and file_units == "feet":
        clip_radius = radius * 3.28084
        print(f"\nRadius: {radius}m = {clip_radius:.2f}ft")
    elif radius_units == "feet" and file_units == "meters":
        clip_radius = radius / 3.28084
        print(f"\nRadius: {radius}ft = {clip_radius:.2f}m")
    elif radius_units == "same":
        print(f"\nRadius: {clip_radius} (file units)")
    else:
        print(f"\nRadius: {clip_radius} {radius_units}")

    # Check if center point is within data bounds
    x_min, x_max = las.x.min(), las.x.max()
    y_min, y_max = las.y.min(), las.y.max()
    print(f"\nData bounds:")
    print(f"  X: [{x_min:.2f}, {x_max:.2f}]")
    print(f"  Y: [{y_min:.2f}, {y_max:.2f}]")

    if not (x_min <= center_x <= x_max and y_min <= center_y <= y_max):
        print("\n  WARNING: Center point is outside data bounds!")
        print("  Double-check your lat/lon and EPSG code.")

    # Clip
    print("\nClipping...")
    clipped, original_count, clipped_count = clip_to_radius(
        las, center_x, center_y, clip_radius
    )

    print(f"  Original points: {original_count:,}")
    print(f"  Clipped points:  {clipped_count:,}")
    print(f"  Retained: {100 * clipped_count / original_count:.1f}%")

    # Clean up temp file
    temp_output.unlink()

    if clipped_count == 0:
        raise cyclopts.ValidationError(
            "No points within radius. Check your coordinates and radius."
        )

    # Step 4: Save output
    print("\n" + "=" * 60)
    print("STEP 4: Saving output")
    print("=" * 60)

    print(f"\nSaving to {output}...")
    clipped.write(str(output))
    file_size = output.stat().st_size / (1024 * 1024)
    print(f"  File size: {file_size:.1f} MB")

    print("\n" + "=" * 60)
    print("DONE!")
    print("=" * 60)
    print(f"\nOutput: {output}")
    print(f"Points: {clipped_count:,}")


@app.command(name="register")
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
            "--reference",
            "-r",
            help="Reference LAS/LAZ file (NYS LiDAR, output from 'ingest' command)",
        ),
    ],
    output: Annotated[
        Path,
        cyclopts.Parameter(
            "--output",
            "-o",
            help="Output merged LAZ/LAS file path",
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
) -> None:
    """
    Register an iPhone LiDAR scan to NYS reference data and merge.

    The processing pipeline:
    1. Load iPhone scan and reference data
    2. Transform iPhone scan to reference CRS (if different)
    3. Align using ICP (reference ground points → all iPhone points)
    4. Transfer ground classification from reference to iPhone points
    5. Merge, replacing reference points in overlap region with iPhone data
    """
    from pyproj import CRS

    from self_survey.clip_to_radius import get_crs_from_las
    from self_survey.register_iphone import (
        load_and_transform_to_crs,
        extract_ground_points,
        icp_align,
        apply_transform_to_las,
        transfer_ground_classification,
        merge_with_replacement,
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

    iphone_las, iphone_crs, transform_offset = load_and_transform_to_crs(
        str(iphone_scan), ref_crs
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
        iphone_all_points = np.vstack(
            (iphone_las.x, iphone_las.y, iphone_las.z)
        ).T
        print(f"  iPhone points: {len(iphone_all_points):,}")

        # Run ICP: iPhone all points -> Reference ground points
        # Non-ground iPhone points won't find correspondences within icp_distance
        print("\nRunning ICP alignment...")
        print("  (iPhone all points → Reference ground points)")
        transform, icp_info = icp_align(
            source_points=iphone_all_points,
            target_points=ref_ground,
            max_correspondence_distance=icp_distance,
            max_iterations=icp_iterations,
        )

        # Check alignment quality
        if icp_info["fitness"] < 0.1:
            print("\n  WARNING: Low ICP fitness score.")
            print("  This is expected if the iPhone scan has many non-ground points.")
            print("  Check the translation values to verify alignment looks reasonable.")

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


def main() -> None:
    """Entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()
