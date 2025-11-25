"""
NYS Parcel lookup utilities.

This module provides functions to query NYS tax parcel data by SBL
(Section-Block-Lot) identifier and retrieve parcel geometry.

Services used:
- NYS Tax Parcels Public: https://gisservices.its.ny.gov/arcgis/rest/services/NYS_Tax_Parcels_Public/
"""

import json
import urllib.parse
import urllib.request
from typing import Any

__all__ = ["NYSParcelLookup", "ParcelInfo"]

# NYS Tax Parcel service endpoints
PARCEL_SERVICE_URL = "https://gisservices.its.ny.gov/arcgis/rest/services/NYS_Tax_Parcels_Public/FeatureServer/1"

# User agent for requests
USER_AGENT = "Mozilla/5.0 (compatible; NYSParcelLookup/1.0; self-survey)"


class ParcelInfo:
    """Container for parcel information and geometry."""

    def __init__(
        self,
        sbl: str,
        swis: str,
        municipality: str,
        address: str,
        geometry: dict[str, Any],
        centroid: tuple[float, float],
        bounds: tuple[float, float, float, float],
        area_sqft: float,
        attributes: dict[str, Any],
    ):
        """
        Initialize parcel info.

        Args:
            sbl: Section-Block-Lot identifier
            swis: SWIS code (municipality identifier)
            municipality: Municipality name
            address: Street address
            geometry: Raw ESRI geometry (rings in Web Mercator)
            centroid: (lon, lat) center point in WGS84
            bounds: (min_lon, min_lat, max_lon, max_lat) in WGS84
            area_sqft: Approximate area in square feet
            attributes: Full attribute dictionary from service
        """
        self.sbl = sbl
        self.swis = swis
        self.municipality = municipality
        self.address = address
        self.geometry = geometry
        self.centroid = centroid
        self.bounds = bounds
        self.area_sqft = area_sqft
        self.attributes = attributes

    def __repr__(self) -> str:
        return f"ParcelInfo(sbl={self.sbl!r}, municipality={self.municipality!r})"


class NYSParcelLookup:
    """
    Query NYS tax parcel data by SBL or other identifiers.

    The NYS Tax Parcels Public service provides parcel boundaries for
    participating counties. Parcels are identified by SBL (Section-Block-Lot)
    combined with SWIS code (municipality identifier).
    """

    def __init__(self, timeout: int = 30):
        """
        Initialize the parcel lookup.

        Args:
            timeout: Request timeout in seconds
        """
        self.timeout = timeout
        self._init_transformers()

    def _init_transformers(self) -> None:
        """Initialize coordinate transformers."""
        from pyproj import CRS, Transformer

        self._wgs84 = CRS.from_epsg(4326)
        self._web_mercator = CRS.from_epsg(3857)
        self._to_wgs84 = Transformer.from_crs(
            self._web_mercator, self._wgs84, always_xy=True
        )
        self._to_web_mercator = Transformer.from_crs(
            self._wgs84, self._web_mercator, always_xy=True
        )

    def _make_request(self, url: str) -> dict[str, Any]:
        """Make an HTTP request and return JSON response."""
        request = urllib.request.Request(url)
        request.add_header("User-Agent", USER_AGENT)

        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            data = json.loads(response.read().decode())

        if "error" in data:
            raise RuntimeError(f"Service error: {data['error']}")

        return data

    def lookup_by_sbl(
        self,
        sbl: str,
        swis: str | None = None,
        municipality: str | None = None,
    ) -> ParcelInfo | None:
        """
        Look up a parcel by SBL identifier.

        Args:
            sbl: Section-Block-Lot identifier (e.g., "123.45-6-7")
            swis: Optional SWIS code to narrow search
            municipality: Optional municipality name to narrow search

        Returns:
            ParcelInfo object if found, None otherwise
        """
        # Build WHERE clause
        where_parts = [f"SBL = '{sbl}'"]

        if swis:
            where_parts.append(f"SWIS = '{swis}'")
        if municipality:
            where_parts.append(f"CITYTOWN_NAME LIKE '%{municipality}%'")

        where_clause = " AND ".join(where_parts)

        return self._query_parcel(where_clause)

    def lookup_by_address(
        self,
        street_number: str,
        street_name: str,
        municipality: str | None = None,
    ) -> ParcelInfo | None:
        """
        Look up a parcel by street address.

        Args:
            street_number: Street number (e.g., "123")
            street_name: Street name (e.g., "Main St")
            municipality: Optional municipality name

        Returns:
            ParcelInfo object if found, None otherwise
        """
        where_parts = [
            f"LOC_ST_NBR = '{street_number}'",
            f"LOC_STREET LIKE '%{street_name.upper()}%'",
        ]

        if municipality:
            where_parts.append(f"CITYTOWN_NAME LIKE '%{municipality}%'")

        where_clause = " AND ".join(where_parts)

        return self._query_parcel(where_clause)

    def lookup_by_swis_printkey(
        self,
        swis: str,
        print_key: str,
    ) -> ParcelInfo | None:
        """
        Look up a parcel by SWIS + Print Key combination.

        This is often the most reliable lookup method as SWIS + Print Key
        uniquely identifies a parcel.

        Args:
            swis: SWIS code (6-digit municipality identifier)
            print_key: Print key (parcel identifier within municipality)

        Returns:
            ParcelInfo object if found, None otherwise
        """
        where_clause = f"SWIS = '{swis}' AND PRINT_KEY = '{print_key}'"
        return self._query_parcel(where_clause)

    def _query_parcel(self, where_clause: str) -> ParcelInfo | None:
        """
        Query the parcel service with a WHERE clause.

        Args:
            where_clause: SQL WHERE clause for filtering

        Returns:
            ParcelInfo object if found, None otherwise
        """
        params = {
            "f": "json",
            "where": where_clause,
            "outFields": "*",
            "returnGeometry": "true",
            "outSR": "3857",
        }

        url = f"{PARCEL_SERVICE_URL}/query?{urllib.parse.urlencode(params)}"

        try:
            data = self._make_request(url)
        except urllib.error.URLError as e:
            raise RuntimeError(f"Failed to query parcel service: {e}") from e

        features = data.get("features", [])
        if not features:
            return None

        # Use first matching feature
        feature = features[0]
        attrs = feature.get("attributes", {})
        geometry = feature.get("geometry", {})

        # Extract key attributes
        sbl = attrs.get("SBL", "")
        swis = attrs.get("SWIS", "")
        municipality = attrs.get("CITYTOWN_NAME", "")
        street_nbr = attrs.get("LOC_ST_NBR", "")
        street = attrs.get("LOC_STREET", "")
        address = f"{street_nbr} {street}".strip()

        # Calculate centroid and bounds from geometry
        centroid, bounds, area_sqft = self._analyze_geometry(geometry)

        return ParcelInfo(
            sbl=sbl,
            swis=swis,
            municipality=municipality,
            address=address,
            geometry=geometry,
            centroid=centroid,
            bounds=bounds,
            area_sqft=area_sqft,
            attributes=attrs,
        )

    def _analyze_geometry(
        self, geometry: dict[str, Any]
    ) -> tuple[tuple[float, float], tuple[float, float, float, float], float]:
        """
        Analyze parcel geometry to extract centroid, bounds, and area.

        Args:
            geometry: ESRI geometry dict with rings in Web Mercator

        Returns:
            Tuple of (centroid, bounds, area_sqft)
            - centroid: (lon, lat) in WGS84
            - bounds: (min_lon, min_lat, max_lon, max_lat) in WGS84
            - area_sqft: Approximate area in square feet
        """
        import numpy as np

        rings = geometry.get("rings", [])
        if not rings:
            return (0.0, 0.0), (0.0, 0.0, 0.0, 0.0), 0.0

        # Collect all points from all rings
        all_points = []
        for ring in rings:
            all_points.extend(ring)

        if not all_points:
            return (0.0, 0.0), (0.0, 0.0, 0.0, 0.0), 0.0

        points = np.array(all_points)

        # Calculate bounds in Web Mercator
        min_x, min_y = points.min(axis=0)
        max_x, max_y = points.max(axis=0)

        # Calculate centroid in Web Mercator
        center_x = (min_x + max_x) / 2
        center_y = (min_y + max_y) / 2

        # Transform to WGS84
        center_lon, center_lat = self._to_wgs84.transform(center_x, center_y)
        min_lon, min_lat = self._to_wgs84.transform(min_x, min_y)
        max_lon, max_lat = self._to_wgs84.transform(max_x, max_y)

        # Calculate approximate area using Shoelace formula on first ring
        # (Web Mercator units are approximately meters at equator)
        outer_ring = np.array(rings[0])
        n = len(outer_ring)
        if n > 2:
            # Shoelace formula
            area_m2 = 0.5 * abs(
                sum(
                    outer_ring[i, 0] * outer_ring[(i + 1) % n, 1]
                    - outer_ring[(i + 1) % n, 0] * outer_ring[i, 1]
                    for i in range(n)
                )
            )
            # Convert to square feet (1 m² ≈ 10.764 ft²)
            area_sqft = area_m2 * 10.764
        else:
            area_sqft = 0.0

        return (
            (center_lon, center_lat),
            (min_lon, min_lat, max_lon, max_lat),
            area_sqft,
        )

    def get_geometry_as_wgs84(
        self, parcel: ParcelInfo
    ) -> list[list[tuple[float, float]]]:
        """
        Get parcel geometry rings transformed to WGS84 (lon, lat).

        Args:
            parcel: ParcelInfo object

        Returns:
            List of rings, each ring is a list of (lon, lat) tuples
        """
        rings = parcel.geometry.get("rings", [])
        wgs84_rings = []

        for ring in rings:
            wgs84_ring = []
            for x, y in ring:
                lon, lat = self._to_wgs84.transform(x, y)
                wgs84_ring.append((lon, lat))
            wgs84_rings.append(wgs84_ring)

        return wgs84_rings

    def get_geometry_as_geojson(self, parcel: ParcelInfo) -> dict[str, Any]:
        """
        Get parcel geometry as GeoJSON Polygon.

        Args:
            parcel: ParcelInfo object

        Returns:
            GeoJSON geometry dict
        """
        wgs84_rings = self.get_geometry_as_wgs84(parcel)

        return {
            "type": "Polygon",
            "coordinates": wgs84_rings,
        }
