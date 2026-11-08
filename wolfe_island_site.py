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

import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import osmnx as ox
import shapely
from shapely.geometry import box

# ---------------------------------------------------------------- config
PLACE = "Wolfe Island, Ontario, Canada"
CRS_M = "EPSG:32618"          # UTM zone 18N: metres, correct for Kingston area

# Optional: restrict to the western part where the real farm sits.
# (min_lon, min_lat, max_lon, max_lat) in WGS84 — set to None for whole island.
SITE_BBOX = (-76.50, 44.08, -76.30, 44.22)   # rough guess: CHECK on the plot

# Setbacks in metres — placeholders, verify against O. Reg. 359/09
SETBACK_HOUSE = 550
SETBACK_ROAD = 65          # blade length + 10 from edge of road
SETBACK_SHORE = 120

GRID_RES = 25                 # raster cell size in metres
OUT = "wolfe_island_site.gpkg"

# ---------------------------------------------------------------- boundary
island = ox.geocode_to_gdf(PLACE)
if island.geometry.iloc[0].geom_type not in ("Polygon", "MultiPolygon"):
    raise ValueError("Geocoder returned a point, not the island polygon. "
                     "Try a different PLACE string or the OSM relation ID.")

if SITE_BBOX is not None:
    island = gpd.clip(island, box(*SITE_BBOX))

site_poly_wgs = island.geometry.union_all()

# ---------------------------------------------------------------- OSM layers
roads = ox.features_from_polygon(site_poly_wgs, tags={"highway": True})
roads = roads[roads.geom_type.isin(["LineString", "MultiLineString"])]

buildings = ox.features_from_polygon(site_poly_wgs, tags={"building": True})
buildings = buildings[buildings.geom_type.isin(["Polygon", "MultiPolygon", "Point"])]

turbines = ox.features_from_polygon(
    site_poly_wgs, tags={"power": "generator", "generator:source": "wind"})
turbines = turbines[turbines.geom_type == "Point"]

# keep only geometry (OSM tag columns are messy) and project to metres
island, roads, buildings, turbines = (
    g[["geometry"]].to_crs(CRS_M) for g in (island, roads, buildings, turbines))

print(f"roads: {len(roads)}  buildings: {len(buildings)}  "
      f"existing turbines: {len(turbines)}")

# ---------------------------------------------------------------- exclusions
site = island.geometry.union_all()

house_zone = buildings.buffer(SETBACK_HOUSE).union_all()
road_zone = roads.buffer(SETBACK_ROAD).union_all()
shore_zone = site.boundary.buffer(SETBACK_SHORE)

excluded = shapely.union_all([house_zone, road_zone, shore_zone])
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

np.savez_compressed("wolfe_island_mask.npz",
                    mask=mask, x0=minx, y0=miny, res=GRID_RES)

# ---------------------------------------------------------------- save
gpd.GeoDataFrame(geometry=[site], crs=CRS_M).to_file(OUT, layer="site")
gpd.GeoDataFrame(geometry=[feasible], crs=CRS_M).to_file(OUT, layer="feasible")
roads.to_file(OUT, layer="roads")
buildings.to_file(OUT, layer="buildings")
if len(turbines):
    turbines.to_file(OUT, layer="existing_turbines")

# ---------------------------------------------------------------- plot
fig, ax = plt.subplots(figsize=(10, 8))
gpd.GeoSeries([site], crs=CRS_M).plot(ax=ax, color="#ddd")
gpd.GeoSeries([feasible], crs=CRS_M).plot(ax=ax, color="#9c9", alpha=0.7)
roads.plot(ax=ax, color="#555", linewidth=0.6)
buildings.centroid.plot(ax=ax, color="crimson", markersize=3)
if len(turbines):
    turbines.plot(ax=ax, color="navy", markersize=12, marker="^")
ax.set_title("Wolfe Island: feasible area (green), roads, houses, turbines")
ax.set_axis_off()
plt.tight_layout()
plt.savefig("wolfe_island_site.png", dpi=200)
plt.show()


# ---------------------------------------------------------------- EA helper
def is_feasible(x, y, mask=mask, x0=minx, y0=miny, res=GRID_RES):
    """O(1) lookup for a candidate turbine at UTM (x, y)."""
    i = int((y - y0) // res)
    j = int((x - x0) // res)
    if 0 <= i < mask.shape[0] and 0 <= j < mask.shape[1]:
        return bool(mask[i, j])
    return False
