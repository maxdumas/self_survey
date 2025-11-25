"""
Download command for NYS LiDAR, DEM, and orthoimagery tiles.

This command queries the NYS GIS services to find tiles intersecting a
specified location and radius, then downloads them to a local directory.

Data sources:
- Orthoimagery: NYS orthos.its.ny.gov ArcGIS REST services
- LiDAR: NYS GIS Clearinghouse FTP (ftp.gis.ny.gov)
- DEM: NYS GIS Clearinghouse FTP (ftp.gis.ny.gov)
"""

from pathlib import Path
from typing import Annotated

import cyclopts


def download(
    lat: Annotated[
        float,
        cyclopts.Parameter(help="Center latitude (WGS84)"),
    ],
    lon: Annotated[
        float,
        cyclopts.Parameter(help="Center longitude (WGS84)"),
    ],
    radius: Annotated[
        float,
        cyclopts.Parameter(help="Search radius in meters"),
    ],
    *,
    output_dir: Annotated[
        Path,
        cyclopts.Parameter(
            name="--output",
            alias="-o",
            help="Output directory for downloaded tiles",
        ),
    ],
    lidar: Annotated[
        bool,
        cyclopts.Parameter(
            name="--lidar",
            alias="-l",
            help="Download LiDAR point cloud tiles (LAZ format)",
        ),
    ] = True,
    dem: Annotated[
        bool,
        cyclopts.Parameter(
            name="--dem",
            alias="-d",
            help="Download DEM tiles (GeoTIFF format)",
        ),
    ] = True,
    ortho: Annotated[
        bool,
        cyclopts.Parameter(
            name="--ortho",
            alias="-i",
            help="Download orthoimagery tiles (GeoTIFF format)",
        ),
    ] = True,
    ortho_year: Annotated[
        str,
        cyclopts.Parameter(
            help="Orthoimagery year to download (e.g., '2022', '2023', or 'latest')",
        ),
    ] = "latest",
    dry_run: Annotated[
        bool,
        cyclopts.Parameter(
            help="List tiles that would be downloaded without downloading",
        ),
    ] = False,
    overwrite: Annotated[
        bool,
        cyclopts.Parameter(
            help="Overwrite existing files",
        ),
    ] = False,
) -> None:
    """
    Download NYS LiDAR, DEM, and orthoimagery tiles for a location.

    Queries the NYS GIS services to find all tiles that intersect with a
    circular buffer around the specified lat/lon point, then downloads
    the matching tiles to the output directory.

    Example:
        preprocess download 42.4532 -73.7891 500 -o ./tiles/

    This will download all available LiDAR, DEM, and orthoimagery tiles
    within 500 meters of the specified point.
    """
    from self_survey.nys_tile_downloader import NYSTileDownloader

    # Create output directory
    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize downloader
    downloader = NYSTileDownloader()

    print("=" * 60)
    print("NYS Tile Downloader")
    print("=" * 60)
    print(f"\nCenter point: ({lat}, {lon})")
    print(f"Search radius: {radius} meters")
    print(f"Output directory: {output_dir}")
    print("\nData types to download:")
    print(f"  LiDAR: {'Yes' if lidar else 'No'}")
    print(f"  DEM: {'Yes' if dem else 'No'}")
    print(f"  Orthoimagery: {'Yes' if ortho else 'No'}")
    if ortho:
        print(f"  Ortho year: {ortho_year}")

    if dry_run:
        print("\n*** DRY RUN - No files will be downloaded ***")

    # Track all tiles to download
    all_tiles: list[dict] = []

    # Query for orthoimagery tiles
    if ortho:
        print("\n" + "-" * 60)
        print("Querying orthoimagery tile index...")
        print("-" * 60)
        ortho_tiles = downloader.query_ortho_tiles(lat, lon, radius, ortho_year)
        print(f"Found {len(ortho_tiles)} orthoimagery tile(s)")
        for tile in ortho_tiles:
            print(f"  - {tile['name']}")
            tile["type"] = "ortho"
            all_tiles.append(tile)

    # Query for LiDAR tiles
    if lidar:
        print("\n" + "-" * 60)
        print("Querying LiDAR tile index...")
        print("-" * 60)
        lidar_tiles = downloader.query_lidar_tiles(lat, lon, radius)
        print(f"Found {len(lidar_tiles)} LiDAR tile(s)")
        for tile in lidar_tiles:
            print(f"  - {tile['name']} ({tile.get('project', 'unknown project')})")
            tile["type"] = "lidar"
            all_tiles.append(tile)

    # Query for DEM tiles
    if dem:
        print("\n" + "-" * 60)
        print("Querying DEM tile index...")
        print("-" * 60)
        dem_tiles = downloader.query_dem_tiles(lat, lon, radius)
        print(f"Found {len(dem_tiles)} DEM tile(s)")
        for tile in dem_tiles:
            print(f"  - {tile['name']}")
            tile["type"] = "dem"
            all_tiles.append(tile)

    if not all_tiles:
        print("\nNo tiles found within the specified radius.")
        print("Possible causes:")
        print("  - No data coverage for this location")
        print("  - NYS GIS services may be temporarily unavailable")
        print("  - Try increasing the radius")
        print("  - Verify the coordinates are within New York State")
        print("\nNote: The NYS GIS services are only accessible from certain networks.")
        print("If you're getting 403 errors, try accessing from a different location.")
        return

    # Summary
    print("\n" + "=" * 60)
    print(f"TOTAL: {len(all_tiles)} tile(s) to download")
    print("=" * 60)

    if dry_run:
        print("\nDry run complete. Use without --dry-run to download.")
        return

    # Download tiles
    print("\nDownloading tiles...")

    downloaded = 0
    skipped = 0
    failed = 0

    for i, tile in enumerate(all_tiles, 1):
        tile_type = tile["type"]
        tile_name = tile["name"]
        url = tile["url"]

        # Determine output subdirectory
        subdir = output_dir / tile_type
        subdir.mkdir(exist_ok=True)

        # Determine output filename
        output_file = subdir / tile_name

        print(f"\n[{i}/{len(all_tiles)}] {tile_name}")

        if output_file.exists() and not overwrite:
            print("  Skipped (already exists)")
            skipped += 1
            continue

        try:
            downloader.download_file(url, output_file)
            file_size = output_file.stat().st_size / (1024 * 1024)
            print(f"  Downloaded ({file_size:.1f} MB)")
            downloaded += 1
        except Exception as e:
            print(f"  Failed: {e}")
            failed += 1

    # Final summary
    print("\n" + "=" * 60)
    print("DOWNLOAD COMPLETE")
    print("=" * 60)
    print(f"  Downloaded: {downloaded}")
    print(f"  Skipped: {skipped}")
    print(f"  Failed: {failed}")
    print(f"\nFiles saved to: {output_dir}")
