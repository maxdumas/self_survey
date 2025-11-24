"""
Command-line interface for self-survey LiDAR processing.

This module provides the main CLI entry point. Commands are defined in
separate modules under the commands/ package.

Usage
-----

    # Ingest NYS LiDAR data
    preprocess ingest tile1.laz tile2.laz --ortho ortho.tif \\
        --lat 42.4532 --lon -73.7891 --radius 200 -o nys_reference.laz

    # Register iPhone scan
    preprocess register polycam_scan.laz --reference nys_reference.laz -o merged.laz

    # Generate elevation contours
    preprocess contour merged.laz -o contours.dxf --interval 2

See README.md for detailed documentation.
"""

import cyclopts

from self_survey.commands.ingest import ingest
from self_survey.commands.register import register
from self_survey.commands.contour import contour

app = cyclopts.App(
    name="preprocess",
    help="Preprocess LiDAR data for property surveys.",
)

# Register commands from separate modules
app.command(ingest, name="ingest")
app.command(register, name="register")
app.command(contour, name="contour")


def main() -> None:
    """Entry point for the CLI."""
    app()


if __name__ == "__main__":
    main()
