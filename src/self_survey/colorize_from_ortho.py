"""
Colorize a point cloud from orthoimagery.

Dependencies:
    pip install laspy[lazrs] numpy rasterio open3d
"""

import numpy as np
import laspy
import rasterio
from rasterio.windows import Window
import open3d as o3d


def colorize_from_ortho(las_path: str, ortho_path: str, output_path: str):
    """
    Add RGB colors to a LAS file by sampling from an orthophoto.
    """
    print(f"Loading {las_path}...")
    las = laspy.read(las_path)
    
    print(f"Opening {ortho_path}...")
    with rasterio.open(ortho_path) as ortho:
        # Get point coordinates
        xs = np.array(las.x)
        ys = np.array(las.y)
        
        # Transform point coordinates to raster pixel coordinates
        rows, cols = rasterio.transform.rowcol(ortho.transform, xs, ys)
        rows = np.array(rows)
        cols = np.array(cols)
        
        # Clamp to image bounds
        rows = np.clip(rows, 0, ortho.height - 1)
        cols = np.clip(cols, 0, ortho.width - 1)
        
        # Read the full image (or tile if huge)
        print("Reading orthoimagery...")
        rgb = ortho.read([1, 2, 3])  # Assumes bands 1,2,3 are RGB
        
        # Sample colors at each point
        print("Sampling colors...")
        red = rgb[0, rows, cols]
        green = rgb[1, rows, cols]
        blue = rgb[2, rows, cols]
    
    # Create new LAS with RGB
    print("Creating colorized point cloud...")
    
    # Need a point format that supports RGB (format 2, 3, 7, or 8)
    new_header = laspy.LasHeader(point_format=3, version="1.4")
    new_header.offsets = las.header.offsets
    new_header.scales = las.header.scales
    
    # Copy VLRs (CRS info)
    for vlr in las.header.vlrs:
        new_header.vlrs.append(vlr)
    
    new_las = laspy.LasData(new_header)
    new_las.x = las.x
    new_las.y = las.y
    new_las.z = las.z
    new_las.intensity = las.intensity
    new_las.classification = las.classification
    
    # LAS stores RGB as 16-bit
    new_las.red = red.astype(np.uint16) * 256
    new_las.green = green.astype(np.uint16) * 256
    new_las.blue = blue.astype(np.uint16) * 256
    
    print(f"Saving to {output_path}...")
    new_las.write(output_path)
    
    print("Done!")
    return new_las


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 4:
        print("Usage: python colorize_from_ortho.py input.laz ortho.tif output.laz")
        sys.exit(1)
    
    colorize_from_ortho(sys.argv[1], sys.argv[2], sys.argv[3])