"""
Build the Wolfe Island site model for wind turbine layout optimization.

Pulls from OpenStreetMap:
  - island boundary
  - roads
  - buildings (noise receptors for setbacks)
  - existing wind turbines (baseline layout)

Then builds buffered exclusion zones, a feasible-area polygon, and a
boolean raster mask the EA can query quickly.

pip install osmnx geopandas shapely matplotlib numpy
"""

from pathlib import Path
import os

import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import osmnx as ox
import shapely
from shapely.geometry import box
from rasterio.features import rasterize
from rasterio.transform import from_origin
from scipy.ndimage import distance_transform_edt, label

# ---------------------------------------------------------------- config
PLACE = "Wolfe Island, Ontario, Canada"
CRS_M = "EPSG:32618"          # UTM zone 18N: metres, correct for Kingston area

# Optional: restrict to the western part where the real farm sits.
# (min_lon, min_lat, max_lon, max_lat) in WGS84 — set to None for whole island.
SITE_BBOX = (-76.50, 44.08, -76.30, 44.20)   # rough guess: CHECK on the plot

# Setbacks in metres — placeholders, verify against O. Reg. 359/09
SETBACK_HOUSE = 550
SETBACK_ROAD = 280          # blade length + 10 from edge of road
SETBACK_SHORE = 120

GRID_RES = 25                 # raster cell size in metres
DATA_DIR = Path(__file__).resolve().parent.parent
OUT = DATA_DIR / "wolfe_island_site.gpkg"
# Use a regional mirror; keep an override for server outages or other sites.
ox.settings.overpass_url = os.environ.get(
    "OSMNX_OVERPASS_URL", "https://ovp.dresel.systems/api-na")
# These mirrors do not use the main server's slot-status protocol. Queries
# here are sequential, and each mirror still enforces its own request limits.
ox.settings.overpass_rate_limit = ox.settings.overpass_url.rstrip("/") not in {
    "https://ovp.dresel.systems/api-na",
    "https://overpass.kumi.systems/api",
    "https://overpass.private.coffee/api",
}
ox.settings.http_user_agent = "EvolutionGroupProject/1.0 (academic wind farm layout modelling)"

# ---------------------------------------------------------------- boundary
print("Downloading island boundary from OpenStreetMap...", flush=True)
island = ox.geocode_to_gdf(PLACE)
if island.geometry.iloc[0].geom_type not in ("Polygon", "MultiPolygon"):
    raise ValueError("Geocoder returned a point, not the island polygon. "
                     "Try a different PLACE string or the OSM relation ID.")

if SITE_BBOX is not None:
    island = gpd.clip(island, box(*SITE_BBOX))

site_poly_wgs = island.geometry.union_all()

# ---------------------------------------------------------------- OSM layers
print("Downloading wetlands, roads, buildings and turbines (first run may take a few minutes)...",
      flush=True)
wetlands = ox.features_from_polygon(
    site_poly_wgs, 
    tags={"natural": "wetland"}
)
wetlands = wetlands[wetlands.geom_type.isin(["Polygon", "MultiPolygon"])]

roads = ox.features_from_polygon(site_poly_wgs, tags={"highway": True})
roads = roads[roads.geom_type.isin(["LineString", "MultiLineString"])]

buildings = ox.features_from_polygon(site_poly_wgs, tags={"building": True})
buildings = buildings[buildings.geom_type.isin(["Polygon", "MultiPolygon", "Point"])]

turbines = ox.features_from_polygon(
    site_poly_wgs, tags={"power": "generator", "generator:source": "wind"})
turbines = turbines[turbines.geom_type == "Point"]

# keep only geometry (OSM tag columns are messy) and project to metres
island = island[["geometry"]].to_crs(CRS_M)
buildings = buildings[["geometry"]].to_crs(CRS_M)
turbines = turbines[["geometry"]].to_crs(CRS_M)
roads = roads[["geometry", "highway"]].to_crs(CRS_M)
wetlands = wetlands[["geometry"]].to_crs(CRS_M)

print(f"roads: {len(roads)}  buildings: {len(buildings)}  "
      f"existing turbines: {len(turbines)}")

# ---------------------------------------------------------------- exclusions
site = island.geometry.union_all()

house_zone = buildings.buffer(SETBACK_HOUSE).union_all()
main_roads = roads[~roads["highway"].isin(["service", "track"])]
service_roads = roads[roads["highway"].isin(["service", "track"])]
road_zone = main_roads.buffer(SETBACK_ROAD).union_all()
shore_zone = site.boundary.buffer(SETBACK_SHORE)

nature_zone = wetlands.geometry.union_all()

excluded = shapely.union_all([house_zone, road_zone, shore_zone, nature_zone])
feasible = site.difference(excluded)

print(f"site area: {site.area/1e6:.1f} km^2   "
      f"feasible area: {feasible.area/1e6:.1f} km^2")

# ---------------------------------------------------------------- raster mask
minx, miny, maxx, maxy = site.bounds
xs = np.arange(minx, maxx, GRID_RES) + GRID_RES / 2
ys = np.arange(miny, maxy, GRID_RES) + GRID_RES / 2
XX, YY = np.meshgrid(xs, ys)
shapely.prepare(feasible)
mask = shapely.contains_xy(feasible, XX, YY)   # True = turbine allowed

# distance (m) from each cell centre to the nearest main road (no service/track)
tf = from_origin(minx, miny + mask.shape[0] * GRID_RES, GRID_RES, GRID_RES)
on_road = rasterize(main_roads.geometry, out_shape=mask.shape, transform=tf,
    all_touched=True, fill=0, default_value = 1, dtype="uint8")
on_road = on_road[::-1]
road_dist = distance_transform_edt(on_road == 0) * GRID_RES

# keep only feasible cells reachable overland from a main road without crossing wetlands
in_wet = (rasterize(wetlands.geometry, out_shape=mask.shape, transform=tf,
                    all_touched=True, fill=0, default_value=1, dtype="uint8")[::-1].astype(bool)
          if len(wetlands) else np.zeros(mask.shape, dtype=bool))
in_site = shapely.contains_xy(site, XX, YY)
passable = in_site & (~in_wet | (on_road == 1))     # roads crossing a wetland still count
labels, _ = label(passable)                          # connected areas of dry land
road_labels = np.unique(labels[(on_road == 1) & passable])
reachable = np.isin(labels, road_labels[road_labels > 0])
print(f"cells cut off by wetland: {(mask & ~reachable).sum()}")
mask &= reachable

habitat_dist = (distance_transform_edt(~in_wet) * GRID_RES if in_wet.any()
                else np.full(mask.shape, np.inf))
np.savez_compressed(DATA_DIR / "wolfe_island_mask.npz", mask=mask, road_dist=road_dist,
                    habitat_dist=habitat_dist, x0=minx, y0=miny, res=GRID_RES)
print(f"Saved site mask: {DATA_DIR / 'wolfe_island_mask.npz'}", flush=True)


# ---------------------------------------------------------------- save
gpd.GeoDataFrame(geometry=[site], crs=CRS_M).to_file(OUT, layer="site")
gpd.GeoDataFrame(geometry=[feasible], crs=CRS_M).to_file(OUT, layer="feasible")
main_roads.to_file(OUT, layer="roads")
if len(wetlands):
    wetlands.to_file(OUT, layer="wetlands")
if len(service_roads):
    service_roads.to_file(OUT, layer="service_roads")
buildings.to_file(OUT, layer="buildings")
if len(turbines):
    turbines.to_file(OUT, layer="existing_turbines")

# ---------------------------------------------------------------- plot
fig, ax = plt.subplots(figsize=(10, 8))
gpd.GeoSeries([site], crs=CRS_M).plot(ax=ax, color="#ddd")
gpd.GeoSeries([feasible], crs=CRS_M).plot(ax=ax, color="#9c9", alpha=0.7)
wetlands.plot(ax=ax, color="darkgreen", alpha=0.6)
main_roads.plot(ax=ax, color="#555", linewidth=0.6)
service_roads = roads[roads["highway"].isin(["service", "track"])]
service_roads.plot(ax=ax, color="brown", linewidth=0.8, linestyle="--")
buildings.centroid.plot(ax=ax, color="crimson", markersize=3)
if len(turbines):
    turbines.plot(ax=ax, color="navy", markersize=12, marker="^")
ax.set_title("Wolfe Island: feasible area (green), roads, houses, turbines")
ax.set_axis_off()
plt.tight_layout()
plt.savefig(DATA_DIR / "wolfe_island_site.png", dpi=200)
plt.show()


# ---------------------------------------------------------------- EA helper
def is_feasible(x, y, mask=mask, x0=minx, y0=miny, res=GRID_RES):
    """O(1) lookup for a candidate turbine at UTM (x, y)."""
    i = int((y - y0) // res)
    j = int((x - x0) // res)
    if 0 <= i < mask.shape[0] and 0 <= j < mask.shape[1]:
        return bool(mask[i, j])
    return False
