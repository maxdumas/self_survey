"""
Parcel command for automated LiDAR processing by NYS parcel ID.

This command combines the functionality of download, ingest, and clip
commands to produce a final point cloud for a specific NYS tax parcel:

1. Look up parcel geometry by SBL (Section-Block-Lot) identifier
2. Download LiDAR, DEM, and orthoimagery tiles intersecting the parcel
3. Merge downloaded tiles into a single point cloud
4. Clip to exact parcel boundary
5. Optionally colorize from orthoimagery

Example Usage
-------------

Process a parcel by SBL:

    preprocess parcel "123.45-6-7" -o parcel_survey.laz

Process with municipality hint (faster lookup):

    preprocess parcel "123.45-6-7" --municipality "Albany" -o survey.laz

Process by SWIS + Print Key (most reliable):

    preprocess parcel --swis 010100 --print-key "123.45-6-7" -o survey.laz
"""

from pathlib import Path
from typing import Annotated

import cyclopts


def parcel(
    sbl: Annotated[
        str | None,
        cyclopts.Parameter(
            help="Section-Block-Lot identifier (e.g., '123.45-6-7')",
        ),
    ] = None,
    *,
    swis: Annotated[
        str | None,
        cyclopts.Parameter(
            name="--swis",
            help="SWIS code (6-digit municipality identifier)",
        ),
    ] = None,
    print_key: Annotated[
        str | None,
        cyclopts.Parameter(
            name="--print-key",
            help="Print key (parcel ID within municipality)",
        ),
    ] = None,
    municipality: Annotated[
        str | None,
        cyclopts.Parameter(
            name="--municipality",
            alias="-m",
            help="Municipality name to narrow search",
        ),
    ] = None,
    output: Annotated[
        Path,
        cyclopts.Parameter(
            name="--output",
            alias="-o",
            help="Output LAZ file path for the final point cloud",
        ),
    ],
    work_dir: Annotated[
        Path | None,
        cyclopts.Parameter(
            name="--work-dir",
            alias="-w",
            help="Working directory for downloaded tiles (default: temp dir)",
        ),
    ] = None,
    colorize: Annotated[
        bool,
        cyclopts.Parameter(
            name="--colorize",
            alias="-c",
            help="Colorize point cloud from orthoimagery",
        ),
    ] = True,
    ortho_year: Annotated[
        str,
        cyclopts.Parameter(
            help="Orthoimagery year for colorization ('2022', '2023', 'latest')",
        ),
    ] = "latest",
    keep_tiles: Annotated[
        bool,
        cyclopts.Parameter(
            help="Keep downloaded tiles after processing (don't delete work_dir)",
        ),
    ] = False,
    buffer_feet: Annotated[
        float,
        cyclopts.Parameter(
            help="Buffer around parcel boundary in feet (default: 0 = exact boundary)",
        ),
    ] = 0.0,
    filter_ground: Annotated[
        bool,
        cyclopts.Parameter(
            help="Keep only ground points in final output",
        ),
    ] = False,
    dry_run: Annotated[
        bool,
        cyclopts.Parameter(
            help="Show parcel info and tiles without downloading",
        ),
    ] = False,
) -> None:
    """
    Process LiDAR data for a NYS tax parcel.

    Looks up a parcel by SBL identifier, downloads all intersecting LiDAR
    and orthoimagery tiles, merges them, and clips to the parcel boundary.

    The output is a single LAZ file containing the point cloud for just
    the specified parcel, optionally colorized from orthoimagery.

    Parcel Identification
    ---------------------

    Parcels can be identified by:

    1. SBL alone: `preprocess parcel "123.45-6-7" -o out.laz`
    2. SBL + municipality: `preprocess parcel "123.45-6-7" -m "Albany" -o out.laz`
    3. SWIS + print key: `preprocess parcel --swis 010100 --print-key "123.45-6-7" -o out.laz`

    The SWIS + print key method is most reliable as it uniquely identifies
    a parcel within the state.

    Coverage Notes
    --------------

    NYS tax parcel data is available for participating counties only.
    If your parcel is not found, it may be in a non-participating county.
    See https://gis.ny.gov/parcels for current coverage.
    """
    import shutil
    import tempfile

    import laspy
    import numpy as np

    from self_survey.nys_parcel_lookup import NYSParcelLookup
    from self_survey.nys_tile_downloader import NYSTileDownloader

    # Validate input - need SBL or SWIS+print_key
    if not sbl and not (swis and print_key):
        raise cyclopts.ValidationError(
            "Must provide either SBL or both --swis and --print-key"
        )

    print("=" * 60)
    print("NYS Parcel LiDAR Processor")
    print("=" * 60)

    # Step 1: Look up parcel
    print("\n" + "-" * 60)
    print("STEP 1: Looking up parcel")
    print("-" * 60)

    lookup = NYSParcelLookup()

    if swis and print_key:
        print(f"\nSearching by SWIS={swis}, Print Key={print_key}...")
        parcel_info = lookup.lookup_by_swis_printkey(swis, print_key)
    else:
        print(f"\nSearching by SBL={sbl}...")
        if municipality:
            print(f"  Municipality hint: {municipality}")
        parcel_info = lookup.lookup_by_sbl(sbl, swis=swis, municipality=municipality)

    if not parcel_info:
        raise cyclopts.ValidationError(
            "Parcel not found. This may be in a non-participating county.\n"
            "Check https://gis.ny.gov/parcels for coverage information."
        )

    print("\nParcel found:")
    print(f"  SBL: {parcel_info.sbl}")
    print(f"  SWIS: {parcel_info.swis}")
    print(f"  Municipality: {parcel_info.municipality}")
    print(f"  Address: {parcel_info.address}")
    print(f"  Area: {parcel_info.area_sqft:,.0f} sq ft ({parcel_info.area_sqft / 43560:.2f} acres)")

    center_lon, center_lat = parcel_info.centroid
    print(f"  Centroid: ({center_lat:.6f}, {center_lon:.6f})")

    min_lon, min_lat, max_lon, max_lat = parcel_info.bounds
    print(f"  Bounds: ({min_lat:.6f}, {min_lon:.6f}) to ({max_lat:.6f}, {max_lon:.6f})")

    # Step 2: Query for intersecting tiles
    print("\n" + "-" * 60)
    print("STEP 2: Finding intersecting tiles")
    print("-" * 60)

    downloader = NYSTileDownloader()
    geometry = parcel_info.geometry

    # Query LiDAR tiles
    print("\nQuerying LiDAR tile index...")
    lidar_tiles = downloader.query_lidar_tiles_by_polygon(geometry)
    print(f"  Found {len(lidar_tiles)} LiDAR tile(s)")
    for tile in lidar_tiles:
        print(f"    - {tile['name']} ({tile.get('project', 'unknown')})")

    # Query ortho tiles if colorizing
    ortho_tiles = []
    if colorize:
        print("\nQuerying orthoimagery tile index...")
        ortho_tiles = downloader.query_ortho_tiles_by_polygon(geometry, ortho_year)
        print(f"  Found {len(ortho_tiles)} orthoimagery tile(s)")
        for tile in ortho_tiles:
            print(f"    - {tile['name']}")

    if not lidar_tiles:
        raise cyclopts.ValidationError(
            "No LiDAR tiles found for this parcel.\n"
            "The area may not have LiDAR coverage yet."
        )

    if dry_run:
        print("\n" + "=" * 60)
        print("DRY RUN COMPLETE")
        print("=" * 60)
        print("\nUse without --dry-run to download and process.")
        return

    # Step 3: Download tiles
    print("\n" + "-" * 60)
    print("STEP 3: Downloading tiles")
    print("-" * 60)

    # Set up work directory
    if work_dir:
        work_dir.mkdir(parents=True, exist_ok=True)
        temp_dir = None
    else:
        temp_dir = tempfile.mkdtemp(prefix="parcel_")
        work_dir = Path(temp_dir)

    lidar_dir = work_dir / "lidar"
    ortho_dir = work_dir / "ortho"
    lidar_dir.mkdir(exist_ok=True)
    ortho_dir.mkdir(exist_ok=True)

    print(f"\nWork directory: {work_dir}")

    # Download LiDAR tiles
    downloaded_lidar = []
    for i, tile in enumerate(lidar_tiles, 1):
        tile_path = lidar_dir / tile["name"]
        print(f"\n[{i}/{len(lidar_tiles)}] {tile['name']}")

        if tile_path.exists():
            print("  Already exists, skipping")
            downloaded_lidar.append(tile_path)
            continue

        try:
            downloader.download_file(tile["url"], tile_path)
            file_size = tile_path.stat().st_size / (1024 * 1024)
            print(f"  Downloaded ({file_size:.1f} MB)")
            downloaded_lidar.append(tile_path)
        except Exception as e:
            print(f"  Failed: {e}")

    # Download ortho tiles
    downloaded_ortho = []
    if colorize and ortho_tiles:
        print("\nDownloading orthoimagery...")
        for i, tile in enumerate(ortho_tiles, 1):
            tile_path = ortho_dir / tile["name"]
            print(f"\n[{i}/{len(ortho_tiles)}] {tile['name']}")

            if tile_path.exists():
                print("  Already exists, skipping")
                downloaded_ortho.append(tile_path)
                continue

            try:
                downloader.download_file(tile["url"], tile_path)
                file_size = tile_path.stat().st_size / (1024 * 1024)
                print(f"  Downloaded ({file_size:.1f} MB)")
                downloaded_ortho.append(tile_path)
            except Exception as e:
                print(f"  Failed: {e}")

    if not downloaded_lidar:
        if temp_dir and not keep_tiles:
            shutil.rmtree(temp_dir)
        raise cyclopts.ValidationError("No LiDAR tiles could be downloaded.")

    # Step 4: Merge tiles
    print("\n" + "-" * 60)
    print("STEP 4: Merging LiDAR tiles")
    print("-" * 60)


    from self_survey.merge_lidar_tiles import (
        load_las_to_open3d,
        merge_point_clouds,
        save_as_laz,
    )

    merged_pcd = None
    merged_meta = None

    for tile_path in downloaded_lidar:
        print(f"\nLoading {tile_path.name}...")
        try:
            pcd, meta = load_las_to_open3d(str(tile_path))
            print(f"  Points: {len(pcd.points):,}")

            if merged_pcd is None:
                merged_pcd = pcd
                merged_meta = meta
            else:
                merged_pcd, merged_meta = merge_point_clouds(
                    merged_pcd, merged_meta, pcd, meta
                )
        except Exception as e:
            print(f"  Error loading: {e}")
            continue

    if merged_pcd is None:
        if temp_dir and not keep_tiles:
            shutil.rmtree(temp_dir)
        raise cyclopts.ValidationError("Failed to load any LiDAR tiles.")

    print(f"\nMerged point cloud: {len(merged_pcd.points):,} points")

    # Step 5: Colorize from orthoimagery (optional)
    if colorize and downloaded_ortho:
        print("\n" + "-" * 60)
        print("STEP 5: Colorizing from orthoimagery")
        print("-" * 60)

        from self_survey.colorize_from_ortho import colorize_point_cloud

        for ortho_path in downloaded_ortho:
            print(f"\nApplying {ortho_path.name}...")
            try:
                merged_pcd = colorize_point_cloud(merged_pcd, str(ortho_path))
            except Exception as e:
                print(f"  Error: {e}")

    # Step 6: Clip to parcel boundary
    print("\n" + "-" * 60)
    print("STEP 6: Clipping to parcel boundary")
    print("-" * 60)

    # Save merged to temp file for CRS-aware clipping
    temp_merged = work_dir / "merged_temp.laz"
    save_as_laz(merged_pcd, merged_meta, str(temp_merged))

    # Reload for clipping
    las = laspy.read(str(temp_merged))

    # Get CRS from file
    from pyproj import CRS as ProjCRS

    from self_survey.clip_to_radius import get_crs_from_las

    crs = get_crs_from_las(las)
    if not crs:
        # Default to NY State Plane East if not detected
        print("  Warning: Could not detect CRS, assuming EPSG:2261 (NY State Plane East)")
        crs = ProjCRS.from_epsg(2261)

    print(f"\nFile CRS: {crs.name}")

    # Transform parcel geometry to file CRS
    from pyproj import Transformer

    web_mercator = ProjCRS.from_epsg(3857)
    transformer = Transformer.from_crs(web_mercator, crs, always_xy=True)

    # Get parcel rings and transform to file CRS
    rings = parcel_info.geometry.get("rings", [])
    if not rings:
        if temp_dir and not keep_tiles:
            shutil.rmtree(temp_dir)
        raise cyclopts.ValidationError("Parcel has no geometry rings.")

    # Transform outer ring to file CRS
    outer_ring_wm = rings[0]
    outer_ring_crs = []
    for x, y in outer_ring_wm:
        tx, ty = transformer.transform(x, y)
        outer_ring_crs.append((tx, ty))

    outer_ring_crs = np.array(outer_ring_crs)

    # Apply buffer if requested
    if buffer_feet != 0:
        print(f"\nApplying {buffer_feet} ft buffer...")
        # Simple buffer by expanding/contracting centroid
        center = outer_ring_crs.mean(axis=0)
        if buffer_feet > 0:
            # Expand outward
            scale = 1.0 + (buffer_feet / np.sqrt(parcel_info.area_sqft))
        else:
            # Contract inward
            scale = 1.0 + (buffer_feet / np.sqrt(parcel_info.area_sqft))
        outer_ring_crs = center + (outer_ring_crs - center) * scale

    # Clip points to polygon using point-in-polygon test
    print("\nClipping to parcel boundary...")
    points = np.vstack((las.x, las.y)).T
    original_count = len(points)

    # Point-in-polygon using ray casting
    from self_survey.polygon_clip import points_in_polygon

    mask = points_in_polygon(points, outer_ring_crs)
    clipped_count = np.sum(mask)

    print(f"  Original points: {original_count:,}")
    print(f"  Clipped points: {clipped_count:,}")
    print(f"  Retained: {100 * clipped_count / original_count:.1f}%")

    if clipped_count == 0:
        if temp_dir and not keep_tiles:
            shutil.rmtree(temp_dir)
        raise cyclopts.ValidationError(
            "No points within parcel boundary. Check CRS alignment."
        )

    # Create clipped LAS
    clipped = las[mask]

    # Optional: filter to ground only
    if filter_ground:
        print("\nFiltering to ground points only...")
        ground_mask = clipped.classification == 2
        ground_count = np.sum(ground_mask)
        if ground_count > 0:
            clipped = clipped[ground_mask]
            print(f"  Ground points: {ground_count:,}")
        else:
            print("  Warning: No ground points found, keeping all points")

    # Clean up temp merged file
    temp_merged.unlink()

    # Step 7: Save output
    print("\n" + "-" * 60)
    print("STEP 7: Saving output")
    print("-" * 60)

    print(f"\nSaving to {output}...")
    output.parent.mkdir(parents=True, exist_ok=True)
    clipped.write(str(output))

    file_size = output.stat().st_size / (1024 * 1024)
    print(f"  File size: {file_size:.1f} MB")
    print(f"  Points: {len(clipped.points):,}")

    # Clean up work directory if using temp
    if temp_dir and not keep_tiles:
        print("\nCleaning up temporary files...")
        shutil.rmtree(temp_dir)
    elif keep_tiles:
        print(f"\nTiles preserved in: {work_dir}")

    print("\n" + "=" * 60)
    print("DONE!")
    print("=" * 60)
    print(f"\nOutput: {output}")
    print(f"Parcel: {parcel_info.sbl} ({parcel_info.municipality})")
    print(f"Points: {len(clipped.points):,}")
