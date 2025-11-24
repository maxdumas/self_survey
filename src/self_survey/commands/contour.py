"""
Contour command for generating elevation contours from LiDAR data.

This command generates elevation contour lines from a point cloud with
ground classification, outputting to DXF format for use in CAD software.

Typical Usage
-------------

Generate 2-foot contours (matching typical survey standards):

    preprocess contour merged_survey.laz -o contours.dxf --interval 2

Generate high-fidelity 0.5-foot contours:

    preprocess contour merged_survey.laz -o contours.dxf --interval 0.5 --resolution 0.5

Output Formats
--------------

**DXF** (recommended): Universal CAD interchange format. Opens in AutoCAD,
SketchUp, Rhino, Revit, and virtually any CAD software. Contours are organized
into CONTOUR_MAJOR and CONTOUR_MINOR layers with appropriate colors and
lineweights.

**GeoJSON**: For GIS workflows or web mapping. Contains elevation as a property
on each LineString feature.
"""

from pathlib import Path
from typing import Annotated

import cyclopts
import laspy


def contour(
    input_file: Annotated[
        Path,
        cyclopts.Parameter(
            help="Input LAS/LAZ file with ground classification",
        ),
    ],
    *,
    output: Annotated[
        Path,
        cyclopts.Parameter(
            "--output",
            "-o",
            help="Output file path (.dxf or .geojson)",
        ),
    ],
    interval: Annotated[
        float,
        cyclopts.Parameter(
            "--interval",
            "-i",
            help="Elevation interval between contours (in file units, typically feet)",
        ),
    ] = 2.0,
    resolution: Annotated[
        float,
        cyclopts.Parameter(
            help="Grid resolution for DEM interpolation (in file units). "
            "Smaller = smoother contours but slower processing.",
        ),
    ] = 1.0,
    index_interval: Annotated[
        int,
        cyclopts.Parameter(
            help="Every Nth contour is a major (index) contour. "
            "E.g., 5 means every 5th contour is major (10ft intervals for 2ft contours).",
        ),
    ] = 5,
    ground_class: Annotated[
        int,
        cyclopts.Parameter(
            help="Classification code for ground points (ASPRS standard: 2)",
        ),
    ] = 2,
    smoothing: Annotated[
        int,
        cyclopts.Parameter(
            help="Gaussian smoothing passes on DEM before contouring. "
            "0 = no smoothing, 1-3 = light to moderate smoothing.",
        ),
    ] = 1,
) -> None:
    """
    Generate elevation contours from ground-classified LiDAR data.

    This command extracts ground points from a LAS/LAZ file, interpolates
    them to a regular grid (DEM), and generates contour lines at the
    specified elevation interval.

    The output DXF file contains contours organized into two layers:
    - CONTOUR_MAJOR: Index contours (every Nth contour), shown in red
    - CONTOUR_MINOR: Intermediate contours, shown in green

    For best results with iPhone-augmented data, use a finer interval
    (0.5-1.0 ft) to capture the additional detail from the higher-density
    point cloud.
    """
    from scipy.ndimage import gaussian_filter

    from self_survey.contour_generation import (
        extract_ground_points,
        create_dem,
        generate_contours,
        export_to_dxf,
        export_to_shapefile,
    )

    # Validate inputs
    if not input_file.exists():
        raise cyclopts.ValidationError(f"Input file not found: {input_file}")

    output_suffix = output.suffix.lower()
    if output_suffix not in (".dxf", ".geojson", ".json"):
        raise cyclopts.ValidationError(
            f"Unsupported output format: {output_suffix}. Use .dxf or .geojson"
        )

    if interval <= 0:
        raise cyclopts.ValidationError("Interval must be positive")

    if resolution <= 0:
        raise cyclopts.ValidationError("Resolution must be positive")

    # Step 1: Load and extract ground points
    print("=" * 60)
    print("STEP 1: Loading and extracting ground points")
    print("=" * 60)

    print(f"\nLoading {input_file}...")
    las = laspy.read(str(input_file))
    print(f"  Total points: {len(las.points):,}")

    ground_points = extract_ground_points(las, classification=ground_class)
    print(f"  Ground points (class={ground_class}): {len(ground_points):,}")

    if len(ground_points) < 100:
        raise cyclopts.ValidationError(
            f"Only {len(ground_points)} ground points found. "
            "Ensure the input file has ground classification."
        )

    # Report elevation range
    z_min, z_max = ground_points[:, 2].min(), ground_points[:, 2].max()
    print(f"  Elevation range: {z_min:.2f} to {z_max:.2f}")
    print(f"  Elevation span: {z_max - z_min:.2f}")

    # Step 2: Create DEM
    print("\n" + "=" * 60)
    print("STEP 2: Creating Digital Elevation Model (DEM)")
    print("=" * 60)

    print(f"\nInterpolating to {resolution}-unit grid...")
    grid_x, grid_y, grid_z = create_dem(ground_points, resolution=resolution)
    print(f"  Grid size: {grid_z.shape[1]} x {grid_z.shape[0]} cells")
    print(f"  Valid cells: {(~np.isnan(grid_z)).sum():,} / {grid_z.size:,}")

    # Optional smoothing
    if smoothing > 0:
        print(f"\nApplying Gaussian smoothing ({smoothing} pass(es))...")
        import numpy as np

        # Replace NaN with local mean for smoothing, then restore NaN
        nan_mask = np.isnan(grid_z)
        grid_z_filled = np.where(nan_mask, np.nanmean(grid_z), grid_z)

        for _ in range(smoothing):
            grid_z_filled = gaussian_filter(grid_z_filled, sigma=1.0)

        grid_z = np.where(nan_mask, np.nan, grid_z_filled)

    # Step 3: Generate contours
    print("\n" + "=" * 60)
    print("STEP 3: Generating contour lines")
    print("=" * 60)

    print(f"\nContour interval: {interval}")
    print(f"Index interval: every {index_interval} contours")

    contours = generate_contours(grid_x, grid_y, grid_z, interval=interval)
    print(f"  Generated {len(contours)} contour levels")

    total_polylines = sum(len(polylines) for _, polylines in contours)
    print(f"  Total polylines: {total_polylines:,}")

    # Step 4: Export
    print("\n" + "=" * 60)
    print("STEP 4: Exporting contours")
    print("=" * 60)

    print(f"\nSaving to {output}...")

    if output_suffix == ".dxf":
        stats = export_to_dxf(
            contours,
            str(output),
            index_interval=index_interval,
        )
        print(f"  Major contours: {stats['major_contours']}")
        print(f"  Minor contours: {stats['minor_contours']}")
        print(f"  Total polylines: {stats['total_polylines']:,}")
        print(f"  Total vertices: {stats['total_vertices']:,}")
    else:
        # GeoJSON
        stats = export_to_shapefile(contours, str(output))
        print(f"  Total contour levels: {stats['total_contours']}")
        print(f"  Total polylines: {stats['total_polylines']:,}")

    print(f"  Elevation range: {stats['elevation_min']:.2f} to {stats['elevation_max']:.2f}")

    file_size = output.stat().st_size / 1024
    print(f"  File size: {file_size:.1f} KB")

    print("\n" + "=" * 60)
    print("DONE!")
    print("=" * 60)
    print(f"\nOutput: {output}")
    print(f"Contour interval: {interval}")
    print(f"Total contour lines: {total_polylines:,}")


# Import numpy at module level for use in command
import numpy as np
