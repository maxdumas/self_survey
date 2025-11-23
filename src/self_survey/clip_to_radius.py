"""
Clip a LAZ/LAS file to a radius around a latitude/longitude point.

Handles coordinate transformation from WGS84 (lat/lon) to whatever
CRS the LAZ file is in (typically State Plane for NYS data).

Dependencies:
    pip install open3d laspy[lazrs] numpy pyproj

Usage:
    python clip_to_radius.py input.laz --lat 42.4532 --lon -73.7891 --radius 100 -o clipped.laz
"""

import numpy as np
import laspy
import open3d as o3d
from pyproj import CRS, Transformer
from pathlib import Path
import argparse


def get_crs_from_las(las: laspy.LasData) -> CRS | None:
    """
    Extract CRS from LAS file VLRs.
    Returns None if CRS cannot be determined.
    """
    # Try the built-in parser first (works for many LAS 1.4 files)
    try:
        crs = las.header.parse_crs()
        if crs:
            return CRS.from_wkt(crs.to_wkt())
    except Exception:
        pass
    
    # Check VLRs manually for WKT or GeoTIFF keys
    for vlr in las.header.vlrs:
        # WKT stored in OGC WKT VLR
        if vlr.user_id == "LASF_Projection" and vlr.record_id == 2111:
            wkt = vlr.record_data.decode('utf-8', errors='ignore').strip('\x00')
            return CRS.from_wkt(wkt)
        
        # Also check for EPSG in description (some files store it there)
        if "EPSG" in str(vlr.description):
            try:
                epsg = int(''.join(filter(str.isdigit, vlr.description)))
                return CRS.from_epsg(epsg)
            except Exception:
                pass
    
    return None


def transform_latlon_to_crs(lat: float, lon: float, target_crs: CRS) -> tuple[float, float]:
    """
    Transform a WGS84 lat/lon point to the target CRS.
    Returns (x, y) in target CRS units.
    """
    wgs84 = CRS.from_epsg(4326)
    transformer = Transformer.from_crs(wgs84, target_crs, always_xy=True)
    
    # Note: Transformer expects (lon, lat) order with always_xy=True
    x, y = transformer.transform(lon, lat)
    return x, y


def clip_to_radius(
    las: laspy.LasData,
    center_x: float,
    center_y: float,
    radius: float,
    radius_units: str = "same"  # "same" (as file), "meters", or "feet"
) -> tuple[laspy.LasData, int, int]:
    """
    Clip LAS points to within radius of center point.
    
    Returns: (clipped_las, original_count, clipped_count)
    """
    points = np.vstack((las.x, las.y)).T
    original_count = len(points)
    
    # Calculate distances from center
    distances = np.sqrt(
        (points[:, 0] - center_x) ** 2 + 
        (points[:, 1] - center_y) ** 2
    )
    
    # Create mask for points within radius
    mask = distances <= radius
    clipped_count = np.sum(mask)
    
    # Create new LAS with only the points within radius
    clipped = las[mask]
    
    return clipped, original_count, clipped_count


def main():
    parser = argparse.ArgumentParser(
        description="Clip LAZ/LAS to radius around a lat/lon point",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Clip to 100-meter radius (auto-converts if file is in feet)
    python clip_to_radius.py input.laz --lat 42.4532 --lon -73.7891 --radius 100 -o clipped.laz
    
    # Clip to 500-foot radius, specifying units explicitly
    python clip_to_radius.py input.laz --lat 42.4532 --lon -73.7891 --radius 500 --radius-units feet -o clipped.laz
    
    # Manually specify CRS if auto-detection fails (NY State Plane East, feet)
    python clip_to_radius.py input.laz --lat 42.4532 --lon -73.7891 --radius 100 --epsg 2261 -o clipped.laz
        """
    )
    
    parser.add_argument("input", help="Input LAZ/LAS file")
    parser.add_argument("--lat", type=float, required=True, help="Center latitude (WGS84)")
    parser.add_argument("--lon", type=float, required=True, help="Center longitude (WGS84)")
    parser.add_argument("--radius", type=float, required=True, help="Clip radius")
    parser.add_argument("--radius-units", choices=["meters", "feet", "same"], default="meters",
                        help="Units for radius (default: meters, converted to file units)")
    parser.add_argument("-o", "--output", required=True, help="Output LAZ/LAS file")
    parser.add_argument("--epsg", type=int, help="EPSG code if auto-detection fails (e.g., 2261 for NY State Plane East)")
    parser.add_argument("--visualize", action="store_true", help="Show result in 3D viewer")
    
    args = parser.parse_args()
    
    # Load input file
    print(f"Loading {args.input}...")
    las = laspy.read(args.input)
    print(f"  Points: {len(las.points):,}")
    
    # Determine CRS
    if args.epsg:
        crs = CRS.from_epsg(args.epsg)
        print(f"  Using specified EPSG:{args.epsg}")
    else:
        crs = get_crs_from_las(las)
        if crs:
            print(f"  Detected CRS: {crs.name}")
        else:
            print("\n  ERROR: Could not detect CRS from file.")
            print("  Please specify --epsg manually.")
            print("  Common NY State Plane codes:")
            print("    EPSG:2261 - NY State Plane East (NAD83, feet)")
            print("    EPSG:2262 - NY State Plane Central (NAD83, feet)")
            print("    EPSG:2263 - NY State Plane West (NAD83, feet)")
            print("    EPSG:32618 - UTM Zone 18N (NAD83, meters)")
            return
    
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
    center_x, center_y = transform_latlon_to_crs(args.lat, args.lon, crs)
    print(f"\nCenter point:")
    print(f"  WGS84: ({args.lat}, {args.lon})")
    print(f"  File CRS: ({center_x:.2f}, {center_y:.2f})")
    
    # Convert radius to file units
    radius = args.radius
    if args.radius_units == "meters" and file_units == "feet":
        radius = args.radius * 3.28084
        print(f"\nRadius: {args.radius}m = {radius:.2f}ft")
    elif args.radius_units == "feet" and file_units == "meters":
        radius = args.radius / 3.28084
        print(f"\nRadius: {args.radius}ft = {radius:.2f}m")
    elif args.radius_units == "same":
        print(f"\nRadius: {radius} (file units)")
    else:
        print(f"\nRadius: {radius} {args.radius_units}")
    
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
    clipped, original_count, clipped_count = clip_to_radius(las, center_x, center_y, radius)
    
    print(f"  Original points: {original_count:,}")
    print(f"  Clipped points:  {clipped_count:,}")
    print(f"  Retained: {100 * clipped_count / original_count:.1f}%")
    
    if clipped_count == 0:
        print("\n  ERROR: No points within radius. Check your coordinates.")
        return
    
    # Save
    print(f"\nSaving to {args.output}...")
    clipped.write(args.output)
    file_size = Path(args.output).stat().st_size / (1024 * 1024)
    print(f"  File size: {file_size:.1f} MB")
    
    # Visualize
    if args.visualize:
        print("\nVisualizing...")
        points = np.vstack((clipped.x, clipped.y, clipped.z)).T
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        
        # Color by elevation
        z = points[:, 2]
        z_norm = (z - z.min()) / (z.max() - z.min() + 1e-8)
        colors = np.column_stack([z_norm, 0.3 * np.ones_like(z_norm), 1 - z_norm])
        pcd.colors = o3d.utility.Vector3dVector(colors)
        
        # Add a marker at the center point
        center_z = clipped.z.mean()  # Approximate center height
        center_marker = o3d.geometry.TriangleMesh.create_sphere(radius=radius * 0.02)
        center_marker.translate([center_x, center_y, center_z])
        center_marker.paint_uniform_color([1, 0, 0])  # Red
        
        o3d.visualization.draw_geometries(
            [pcd, center_marker],
            window_name=f"Clipped to {args.radius}{args.radius_units} radius",
            width=1400,
            height=900
        )
    
    print("\nDone!")


if __name__ == "__main__":
    main()