"""
NYS GIS tile index query and download utilities.

This module provides functions to query NYS GIS services for tile indexes
and download LiDAR, DEM, and orthoimagery tiles.

Services used:
- Orthoimagery index: https://orthos.its.ny.gov/arcgis/rest/services/vector/ortho_indexes/
- LiDAR index: https://orthos.its.ny.gov/arcgis/rest/services/vector/las_indexes/
- DEM: NYS GIS Clearinghouse
"""

import json
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from pyproj import CRS, Transformer

__all__ = ["NYSTileDownloader"]

# NYS ArcGIS REST service endpoints (from orthos.its.ny.gov)
ORTHO_INDEX_URL = "https://orthos.its.ny.gov/arcgis/rest/services/vector/ortho_indexes/MapServer"
LIDAR_INDEX_URL = "https://orthos.its.ny.gov/arcgis/rest/services/vector/las_indexes/MapServer"

# FTP/HTTP base URLs for downloads
FTP_LIDAR_BASE = "ftp://ftp.gis.ny.gov/elevation/LIDAR"
FTP_DEM_BASE = "ftp://ftp.gis.ny.gov/elevation/DEM"
FTP_ORTHO_BASE = "ftp://ftp.gis.ny.gov/orthos"

# User agent for requests
USER_AGENT = "Mozilla/5.0 (compatible; NYSTileDownloader/1.0; self-survey)"


class NYSTileDownloader:
    """
    Query NYS GIS services for tile indexes and download tiles.

    This class handles:
    1. Querying ArcGIS REST services with point+buffer geometry
    2. Parsing tile metadata including download URLs
    3. Downloading tiles from FTP/HTTPS sources
    """

    def __init__(self, timeout: int = 30):
        """
        Initialize the downloader.

        Args:
            timeout: Request timeout in seconds
        """
        self.timeout = timeout
        # WGS84 for lat/lon input
        self._wgs84 = CRS.from_epsg(4326)
        # Web Mercator for ArcGIS queries
        self._web_mercator = CRS.from_epsg(3857)
        self._transformer = Transformer.from_crs(
            self._wgs84, self._web_mercator, always_xy=True
        )

    def _transform_to_web_mercator(
        self, lat: float, lon: float
    ) -> tuple[float, float]:
        """Transform WGS84 lat/lon to Web Mercator coordinates."""
        x, y = self._transformer.transform(lon, lat)
        return x, y

    def _query_arcgis_service(
        self,
        base_url: str,
        layer_id: int,
        lat: float,
        lon: float,
        radius_meters: float,
        out_fields: str = "*",
    ) -> list[dict[str, Any]]:
        """
        Query an ArcGIS MapServer or FeatureServer layer.

        Args:
            base_url: Base URL of the service
            layer_id: Layer ID to query
            lat: Center latitude (WGS84)
            lon: Center longitude (WGS84)
            radius_meters: Search radius in meters
            out_fields: Fields to return (default: all)

        Returns:
            List of feature attributes
        """
        import json

        # Transform center point to Web Mercator
        center_x, center_y = self._transform_to_web_mercator(lat, lon)

        # Build query URL
        query_url = f"{base_url}/{layer_id}/query"

        params = {
            "f": "json",
            "geometry": f"{center_x},{center_y}",
            "geometryType": "esriGeometryPoint",
            "inSR": "3857",
            "spatialRel": "esriSpatialRelIntersects",
            "distance": str(radius_meters),
            "units": "esriSRUnit_Meter",
            "outFields": out_fields,
            "returnGeometry": "false",
        }

        url = f"{query_url}?{urllib.parse.urlencode(params)}"

        try:
            request = urllib.request.Request(url)
            request.add_header("User-Agent", USER_AGENT)
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                data = json.loads(response.read().decode())

            if "error" in data:
                raise RuntimeError(f"ArcGIS query error: {data['error']}")

            features = data.get("features", [])
            return [f.get("attributes", {}) for f in features]

        except urllib.error.URLError as e:
            raise RuntimeError(f"Failed to query service: {e}") from e

    def _query_feature_service(
        self,
        service_url: str,
        lat: float,
        lon: float,
        radius_meters: float,
        out_fields: str = "*",
    ) -> list[dict[str, Any]]:
        """
        Query an ArcGIS FeatureServer directly.

        Args:
            service_url: Full URL to the feature layer (ending in /0, /1, etc.)
            lat: Center latitude (WGS84)
            lon: Center longitude (WGS84)
            radius_meters: Search radius in meters
            out_fields: Fields to return

        Returns:
            List of feature attributes
        """
        import json

        center_x, center_y = self._transform_to_web_mercator(lat, lon)

        query_url = f"{service_url}/query"

        params = {
            "f": "json",
            "geometry": f"{center_x},{center_y}",
            "geometryType": "esriGeometryPoint",
            "inSR": "3857",
            "spatialRel": "esriSpatialRelIntersects",
            "distance": str(radius_meters),
            "units": "esriSRUnit_Meter",
            "outFields": out_fields,
            "returnGeometry": "false",
        }

        url = f"{query_url}?{urllib.parse.urlencode(params)}"

        try:
            request = urllib.request.Request(url)
            request.add_header("User-Agent", USER_AGENT)
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                data = json.loads(response.read().decode())

            if "error" in data:
                raise RuntimeError(f"Feature service query error: {data['error']}")

            features = data.get("features", [])
            return [f.get("attributes", {}) for f in features]

        except urllib.error.URLError as e:
            raise RuntimeError(f"Failed to query feature service: {e}") from e

    def _query_arcgis_service_by_polygon(
        self,
        base_url: str,
        layer_id: int,
        geometry: dict[str, Any],
        out_fields: str = "*",
    ) -> list[dict[str, Any]]:
        """
        Query an ArcGIS MapServer layer using polygon geometry.

        Args:
            base_url: Base URL of the service
            layer_id: Layer ID to query
            geometry: ESRI geometry dict (polygon with rings in Web Mercator)
            out_fields: Fields to return (default: all)

        Returns:
            List of feature attributes
        """
        query_url = f"{base_url}/{layer_id}/query"

        # Convert geometry to JSON string for the request
        geometry_json = json.dumps(geometry)

        params = {
            "f": "json",
            "geometry": geometry_json,
            "geometryType": "esriGeometryPolygon",
            "inSR": "3857",
            "spatialRel": "esriSpatialRelIntersects",
            "outFields": out_fields,
            "returnGeometry": "false",
        }

        url = f"{query_url}?{urllib.parse.urlencode(params)}"

        try:
            request = urllib.request.Request(url)
            request.add_header("User-Agent", USER_AGENT)
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                data = json.loads(response.read().decode())

            if "error" in data:
                raise RuntimeError(f"ArcGIS query error: {data['error']}")

            features = data.get("features", [])
            return [f.get("attributes", {}) for f in features]

        except urllib.error.URLError as e:
            raise RuntimeError(f"Failed to query service: {e}") from e

    def query_ortho_tiles_by_polygon(
        self,
        geometry: dict[str, Any],
        year: str = "latest",
    ) -> list[dict[str, Any]]:
        """
        Query orthoimagery tiles intersecting a polygon.

        Args:
            geometry: ESRI geometry dict (polygon with rings in Web Mercator)
            year: Imagery year ('2021', '2022', '2023', 'latest')

        Returns:
            List of tile info dicts with 'name' and 'url' keys
        """
        year_layer_map = {
            "latest": 0,
            "2024": 1,
            "2023": 2,
            "2022": 3,
            "2021": 4,
            "2020": 5,
            "2019": 6,
            "2018": 7,
        }

        layer_id = year_layer_map.get(year.lower(), 0)

        try:
            features = self._query_arcgis_service_by_polygon(
                ORTHO_INDEX_URL,
                layer_id,
                geometry,
                out_fields="*",
            )
        except Exception as e:
            print(f"  Warning: Could not query ortho index service: {e}")
            return []

        return self._parse_ortho_features(features, year)

    def query_lidar_tiles_by_polygon(
        self,
        geometry: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """
        Query LiDAR tiles intersecting a polygon.

        Args:
            geometry: ESRI geometry dict (polygon with rings in Web Mercator)

        Returns:
            List of tile info dicts with 'name', 'url', and 'project' keys
        """
        try:
            features = self._query_arcgis_service_by_polygon(
                LIDAR_INDEX_URL,
                0,
                geometry,
                out_fields="*",
            )
        except Exception as e:
            print(f"  Warning: Could not query LiDAR index service: {e}")
            # Try alternative layers
            for layer_id in range(1, 10):
                try:
                    features = self._query_arcgis_service_by_polygon(
                        LIDAR_INDEX_URL,
                        layer_id,
                        geometry,
                        out_fields="*",
                    )
                    if features:
                        break
                except Exception:
                    continue
            else:
                return []

        return self._parse_lidar_features(features)

    def query_dem_tiles_by_polygon(
        self,
        geometry: dict[str, Any],
    ) -> list[dict[str, Any]]:
        """
        Query DEM tiles intersecting a polygon.

        Args:
            geometry: ESRI geometry dict (polygon with rings in Web Mercator)

        Returns:
            List of tile info dicts with 'name' and 'url' keys
        """
        # Get LiDAR tiles first, then derive DEM info
        lidar_tiles = self.query_lidar_tiles_by_polygon(geometry)

        tiles = []
        seen_projects = set()

        for tile in lidar_tiles:
            project = tile.get("project", "unknown")
            if project != "unknown" and project not in seen_projects:
                seen_projects.add(project)
                dem_name = f"{project}_dem.tif"
                dem_url = f"{FTP_DEM_BASE}/{project}/{dem_name}"
                tiles.append({
                    "name": dem_name,
                    "url": dem_url,
                    "project": project,
                    "attributes": {},
                })

        return tiles

    def _parse_ortho_features(
        self, features: list[dict[str, Any]], year: str
    ) -> list[dict[str, Any]]:
        """Parse ortho feature attributes into tile info dicts."""
        tiles = []
        for attrs in features:
            name = (
                attrs.get("TILENAME")
                or attrs.get("TileName")
                or attrs.get("tile_name")
                or attrs.get("NAME")
                or attrs.get("name")
                or "unknown"
            )

            url = (
                attrs.get("DOWNLOAD")
                or attrs.get("Download")
                or attrs.get("download_url")
                or attrs.get("URL")
                or attrs.get("url")
                or attrs.get("DownloadURL")
            )

            if url:
                if not name.endswith((".tif", ".tiff", ".zip", ".sid")):
                    name = f"{name}.zip"

                tiles.append({
                    "name": name,
                    "url": url,
                    "year": year,
                    "attributes": attrs,
                })

        return tiles

    def _parse_lidar_features(
        self, features: list[dict[str, Any]]
    ) -> list[dict[str, Any]]:
        """Parse LiDAR feature attributes into tile info dicts."""
        tiles = []
        for attrs in features:
            name = (
                attrs.get("TILENAME")
                or attrs.get("TileName")
                or attrs.get("tile_name")
                or attrs.get("NAME")
                or attrs.get("Tile")
                or "unknown"
            )

            project = (
                attrs.get("PROJECT")
                or attrs.get("Project")
                or attrs.get("project_name")
                or attrs.get("Collection")
                or "unknown"
            )

            url = (
                attrs.get("DOWNLOAD")
                or attrs.get("Download")
                or attrs.get("download_url")
                or attrs.get("URL")
                or attrs.get("url")
                or attrs.get("DownloadPath")
            )

            if not url and name != "unknown":
                url = f"{FTP_LIDAR_BASE}/{project}/{name}"
                if not url.endswith((".laz", ".las", ".zip")):
                    url = f"{url}.laz"

            if url:
                if not name.endswith((".laz", ".las", ".zip")):
                    name = f"{name}.laz"

                tiles.append({
                    "name": name,
                    "url": url,
                    "project": project,
                    "attributes": attrs,
                })

        return tiles

    def query_ortho_tiles(
        self,
        lat: float,
        lon: float,
        radius_meters: float,
        year: str = "latest",
    ) -> list[dict[str, Any]]:
        """
        Query orthoimagery tile index for tiles intersecting the search area.

        The NYS orthoimagery index service provides tile boundaries with
        download URLs to the FTP server.

        Args:
            lat: Center latitude (WGS84)
            lon: Center longitude (WGS84)
            radius_meters: Search radius in meters
            year: Imagery year ('2021', '2022', '2023', 'latest')

        Returns:
            List of tile info dicts with 'name' and 'url' keys
        """
        # Map year to layer ID in the ortho_indexes service
        # The service has multiple layers for different years
        year_layer_map = {
            "latest": 0,  # Latest composite
            "2024": 1,
            "2023": 2,
            "2022": 3,
            "2021": 4,
            "2020": 5,
            "2019": 6,
            "2018": 7,
        }

        layer_id = year_layer_map.get(year.lower(), 0)

        try:
            features = self._query_arcgis_service(
                ORTHO_INDEX_URL,
                layer_id,
                lat,
                lon,
                radius_meters,
                out_fields="*",
            )
        except Exception as e:
            print(f"  Warning: Could not query ortho index service: {e}")
            print("  Trying alternative method...")
            return self._query_ortho_tiles_alternative(lat, lon, radius_meters, year)

        tiles = []
        for attrs in features:
            # Extract tile name and download URL from attributes
            # Field names vary by layer, try common patterns
            name = (
                attrs.get("TILENAME")
                or attrs.get("TileName")
                or attrs.get("tile_name")
                or attrs.get("NAME")
                or attrs.get("name")
                or "unknown"
            )

            url = (
                attrs.get("DOWNLOAD")
                or attrs.get("Download")
                or attrs.get("download_url")
                or attrs.get("URL")
                or attrs.get("url")
                or attrs.get("DownloadURL")
            )

            if url:
                # Ensure filename has extension
                if not name.endswith((".tif", ".tiff", ".zip", ".sid")):
                    name = f"{name}.zip"

                tiles.append({
                    "name": name,
                    "url": url,
                    "year": year,
                    "attributes": attrs,
                })

        return tiles

    def _query_ortho_tiles_alternative(
        self,
        _lat: float,
        _lon: float,
        _radius_meters: float,
        _year: str = "latest",
    ) -> list[dict[str, Any]]:
        """
        Alternative method to find orthoimagery tiles using direct FTP structure.

        When the ArcGIS service is unavailable, we can construct likely tile
        names based on the NY State Plane grid.
        """
        # This is a fallback that lists what tiles might exist
        # The NYS ortho tiles are typically named by USNG/MGRS grid cells
        print("  Using alternative tile discovery (limited coverage)")
        return []

    def query_lidar_tiles(
        self,
        lat: float,
        lon: float,
        radius_meters: float,
    ) -> list[dict[str, Any]]:
        """
        Query LiDAR tile index for tiles intersecting the search area.

        Args:
            lat: Center latitude (WGS84)
            lon: Center longitude (WGS84)
            radius_meters: Search radius in meters

        Returns:
            List of tile info dicts with 'name', 'url', and 'project' keys
        """
        # Try the NYS LiDAR index service (las_indexes MapServer)
        # The service has layers for different LiDAR projects/years
        try:
            # Query the main LiDAR index layer (layer 0 typically has all tiles)
            features = self._query_arcgis_service(
                LIDAR_INDEX_URL,
                0,  # Main index layer
                lat,
                lon,
                radius_meters,
                out_fields="*",
            )
        except Exception as e:
            print(f"  Warning: Could not query LiDAR index service: {e}")
            print("  Trying alternative endpoint...")
            return self._query_lidar_tiles_alternative(lat, lon, radius_meters)

        tiles = []
        for attrs in features:
            # Extract tile information
            name = (
                attrs.get("TILENAME")
                or attrs.get("TileName")
                or attrs.get("tile_name")
                or attrs.get("NAME")
                or attrs.get("Tile")
                or "unknown"
            )

            project = (
                attrs.get("PROJECT")
                or attrs.get("Project")
                or attrs.get("project_name")
                or attrs.get("Collection")
                or "unknown"
            )

            url = (
                attrs.get("DOWNLOAD")
                or attrs.get("Download")
                or attrs.get("download_url")
                or attrs.get("URL")
                or attrs.get("url")
                or attrs.get("DownloadPath")
            )

            # If no direct URL, construct from FTP base
            if not url and name != "unknown":
                # Standard path structure: ftp.gis.ny.gov/elevation/LIDAR/{project}/{tile}.laz
                url = f"{FTP_LIDAR_BASE}/{project}/{name}"
                if not url.endswith((".laz", ".las", ".zip")):
                    url = f"{url}.laz"

            if url:
                if not name.endswith((".laz", ".las", ".zip")):
                    name = f"{name}.laz"

                tiles.append({
                    "name": name,
                    "url": url,
                    "project": project,
                    "attributes": attrs,
                })

        return tiles

    def _query_lidar_tiles_alternative(
        self,
        lat: float,
        lon: float,
        radius_meters: float,
    ) -> list[dict[str, Any]]:
        """
        Alternative method: try other layers in the las_indexes service.
        """
        # Try querying other layers (1, 2, etc.) which may have different projects
        for layer_id in range(1, 10):
            try:
                features = self._query_arcgis_service(
                    LIDAR_INDEX_URL,
                    layer_id,
                    lat,
                    lon,
                    radius_meters,
                    out_fields="*",
                )
                if features:
                    tiles = []
                    for attrs in features:
                        name = (
                            attrs.get("TILENAME")
                            or attrs.get("TileName")
                            or attrs.get("tile")
                            or attrs.get("NAME")
                            or "unknown"
                        )

                        project = (
                            attrs.get("PROJECT")
                            or attrs.get("Project")
                            or attrs.get("project_name")
                            or "unknown"
                        )

                        url = (
                            attrs.get("DOWNLOAD")
                            or attrs.get("download")
                            or attrs.get("URL")
                        )

                        if not url and name != "unknown" and project != "unknown":
                            url = f"{FTP_LIDAR_BASE}/{project}/{name}"
                            if not url.endswith((".laz", ".las", ".zip")):
                                url = f"{url}.laz"

                        if url:
                            if not name.endswith((".laz", ".las", ".zip")):
                                name = f"{name}.laz"

                            tiles.append({
                                "name": name,
                                "url": url,
                                "project": project,
                                "attributes": attrs,
                            })
                    if tiles:
                        return tiles
            except Exception:
                continue

        return []

    def query_dem_tiles(
        self,
        lat: float,
        lon: float,
        radius_meters: float,
    ) -> list[dict[str, Any]]:
        """
        Query DEM tile index for tiles intersecting the search area.

        Args:
            lat: Center latitude (WGS84)
            lon: Center longitude (WGS84)
            radius_meters: Search radius in meters

        Returns:
            List of tile info dicts with 'name' and 'url' keys
        """
        # DEM tiles are typically associated with LiDAR projects
        # Try looking for DEM-specific layers in the las_indexes service
        # or construct URLs based on the LiDAR project structure
        try:
            # First try to get LiDAR tiles, then look for associated DEMs
            lidar_tiles = self.query_lidar_tiles(lat, lon, radius_meters)

            # DEM tiles are typically in the same project folder as LiDAR
            # with naming convention like {project}/DEM/{tilename}.tif
            tiles = []
            seen_projects = set()

            for tile in lidar_tiles:
                project = tile.get("project", "unknown")
                if project != "unknown" and project not in seen_projects:
                    seen_projects.add(project)
                    # Construct a potential DEM URL for this project
                    dem_name = f"{project}_dem.tif"
                    dem_url = f"{FTP_DEM_BASE}/{project}/{dem_name}"
                    tiles.append({
                        "name": dem_name,
                        "url": dem_url,
                        "project": project,
                        "attributes": {},
                    })

            return tiles

        except Exception as e:
            print(f"  Warning: Could not query DEM service: {e}")
            return []

    def _query_dem_tiles_alternative(
        self,
        _lat: float,
        _lon: float,
        _radius_meters: float,
    ) -> list[dict[str, Any]]:
        """
        Alternative method - returns empty list as DEMs are derived from LiDAR.
        """
        # DEM tiles are typically derived from LiDAR and stored alongside
        # For now, return empty list as we handle this in query_dem_tiles
        return []

    def download_file(
        self,
        url: str,
        output_path: Path,
    ) -> None:
        """
        Download a file from HTTP/HTTPS/FTP URL.

        Args:
            url: Source URL (HTTP, HTTPS, or FTP)
            output_path: Local path to save the file
        """
        import shutil

        try:
            request = urllib.request.Request(url)
            request.add_header("User-Agent", USER_AGENT)
            with (
                urllib.request.urlopen(request, timeout=self.timeout) as response,
                open(output_path, "wb") as out_file,
            ):
                shutil.copyfileobj(response, out_file)
        except urllib.error.URLError as e:
            # Clean up partial download
            if output_path.exists():
                output_path.unlink()
            raise RuntimeError(f"Download failed: {e}") from e
