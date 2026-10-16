#| title: Oil slicks and ships from Sentinel-1 radar (Python)
#| description: Detects the 2018 Balikpapan oil slick as dark spots on Sentinel-1, separates it from look-alikes, and turns ship detections on twelve-day Sentinel-1 passes over the Malacca Strait into ships per day, with an honest interval and a check against the official count.

# Fullstack Remote Sensing | Code by Rifky Nauval Hendrawan, Remote Sensing Specialist
# MIT licence: free to use and adapt; please keep this credit line.

"""
CHAPTER 77 | Oil and ships under radar

Radar sees the sea through cloud and at night. Two things stand out on it:
oil, which calms the small waves and leaves a dark patch, and ships, which are
steel and shine. This script does both, on public Sentinel-1 data.

Part 1. The Balikpapan Bay oil spill of 31 March 2018: one Sentinel-1 pass on
2 April at 05:50 local time, dark-spot detection against the surrounding sea,
and the features that separate a slick from a look-alike.

Part 2. Ships in the Malacca Strait, 2025: every ascending Sentinel-1 pass that
covers a gate across the strait (29 passes, one every 12 days), a CFAR ship
detector, fixed targets removed, and a traffic-flow estimate of ships per day
compared with the Malaysian Marine Department's STRAITREP count.

The R twin does the same analysis with terra. Both read the same files:
  data/ch77_balikpapan_vv_20180402.tif   VV (dB x 100) and sea mask, 25 m
  data/ch77_malacca_ships_2025.csv       ship detections on every 2025 pass

Environment: pip install earthengine-api numpy pandas scipy rasterio matplotlib
"""

import os
from pathlib import Path

import ee
import numpy as np
import pandas as pd
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[2]
BAY = [116.75, -1.42, 117.15, -1.12]
GATE = [100.2, 2.75, 101.15, 3.05]
STRAITREP_PER_DAY = 281      # Marine Department Malaysia: 102,525 transits in 2025, vessels > 300 GT


# PART 1. The slick -----------------------------------------------------------------------------
def slick_analysis():
    """Dark spots more than 3 dB below the sea within ~10 km, and below -24 dB; patches >= 0.5 km2."""
    import rasterio
    with rasterio.open(ROOT / "data" / "ch77_balikpapan_vv_20180402.tif") as src:
        vv = src.read(1).astype(float) / 100
        sea = src.read(2) == 1
        transform, crs = src.transform, src.crs
    vv[~sea] = np.nan
    valid = (~np.isnan(vv)).astype(float)                      # 5 x 5 mean (125 m) against speckle, ignoring land
    vv_f = ndimage.uniform_filter(np.nan_to_num(vv), 5) / np.maximum(ndimage.uniform_filter(valid, 5), 1e-9)
    vv_f[~sea] = np.nan
    # Background: median of 250 m blocks, then a 41 x 41 block median (about 10 km). A slick can be 10-20 km
    # wide; a small window would sit inside it and call it normal.
    h, w = (vv_f.shape[0] // 10) * 10, (vv_f.shape[1] // 10) * 10
    with np.errstate(all="ignore"):
        blocks = np.nanmedian(vv_f[:h, :w].reshape(h // 10, 10, w // 10, 10), axis=(1, 3))   # land blocks stay NaN
        bg_blocks = ndimage.generic_filter(blocks, np.nanmedian, size=41, mode="nearest", cval=np.nan)  # sea only
    bg = np.kron(bg_blocks, np.ones((10, 10)))
    bg = np.pad(bg, ((0, vv_f.shape[0] - h), (0, vv_f.shape[1] - w)), mode="edge")
    dark = (vv_f < bg - 3) & (vv_f < -24) & sea
    lab, n = ndimage.label(dark, structure=np.ones((3, 3)))
    px_km2 = abs(transform.a * transform.e) / 1e6
    rows = []
    for i, sl in enumerate(ndimage.find_objects(lab), start=1):
        m = lab[sl] == i
        area = m.sum() * px_km2
        if area < 0.5:
            continue
        contrast = float(np.nanmean((bg[sl] - vv_f[sl])[m]))
        perim = (m ^ ndimage.binary_erosion(m)).sum() * abs(transform.a) / 1000
        rows.append({"id": i, "area_km2": area, "contrast_db": contrast, "perimeter_km": perim})
    st = pd.DataFrame(rows)
    st["compactness"] = 4 * np.pi * st.area_km2 / st.perimeter_km ** 2
    st["class"] = np.select([(st.area_km2 >= 5) & (st.contrast_db >= 5), st.contrast_db < 4],
                            ["likely oil", "look-alike: low contrast"], "check: small or weak")
    return vv_f, lab, st.sort_values("area_km2", ascending=False), transform, crs


def slick_table():
    return slick_analysis()[2].round(2)


def plot_slick():
    import matplotlib.pyplot as plt
    from pyproj import Transformer
    vv_f, lab, st, tr, crs = slick_analysis()
    oil = np.isin(lab, st.id[st["class"] == "likely oil"])
    other = np.isin(lab, st.id[st["class"] != "likely oil"])
    ny, nx = vv_f.shape
    t = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
    lon0, lat1 = t.transform(tr.c, tr.f)
    lon1, lat0 = t.transform(tr.c + tr.a * nx, tr.f + tr.e * ny)
    fig, ax = plt.subplots(figsize=(10, 7.5))
    ax.imshow(np.clip(vv_f, -30, -15), cmap="gray", extent=[lon0, lon1, lat0, lat1])
    ax.imshow(np.ma.masked_where(~oil, oil), cmap="autumn", alpha=0.55, extent=[lon0, lon1, lat0, lat1])
    ax.imshow(np.ma.masked_where(~other, other), cmap="cool", alpha=0.8, extent=[lon0, lon1, lat0, lat1])
    ax.set_title(f"Balikpapan Bay, 2 April 2018: {st.area_km2[st['class'] == 'likely oil'].sum():.0f} km² of likely oil (orange)",
                 loc="left", fontweight="bold")
    ax.set_xlabel("Longitude (°E)"); ax.set_ylabel("Latitude (°)")
    ax.grid(color="white", alpha=0.4, lw=0.5)
    return fig


# PART 2. Ships -----------------------------------------------------------------------------------
def ships(min_px=3):
    """Ship detections with fixed targets (lit in more than a quarter of passes, per 500 m cell) removed."""
    d = pd.read_csv(ROOT / "data" / "ch77_malacca_ships_2025.csv", parse_dates=["date"])
    d["cx"], d["cy"] = (d.lon / 0.0045).round(), (d.lat / 0.0045).round()
    n_pass = d.date.nunique()
    lit = d.drop_duplicates(["date", "cx", "cy"]).groupby(["cx", "cy"]).size()
    fixed = set(lit[lit > n_pass / 4].index)
    keep = np.array([(a, b) not in fixed for a, b in zip(d.cx, d.cy)])
    return d[keep & (d.npx >= min_px)], sorted(d.date.unique())


def per_pass(min_px=3):
    m, dates = ships(min_px)
    return m.groupby("date").size().reindex(dates, fill_value=0).rename("ships").reset_index()


L_KM = 33.2 / np.cos(np.radians(45))      # 0.30 deg of latitude, crossed at 135 deg: a 46.9 km path


def flow_table():
    """Ships per day = ships in the gate / path length x speed x 24 h, bootstrapped over passes and speeds."""
    rng = np.random.default_rng(77)
    rows = []
    for k in (3, 5, 10, 20, 40):
        n = per_pass(k).ships.to_numpy()
        sims = [rng.choice(n, n.size).mean() / L_KM * rng.uniform(10, 16) * 1.852 * 24 for _ in range(5000)]
        lo, mid, hi = np.percentile(sims, [5, 50, 95])
        rows.append({"minimum target (pixels)": k, "bright area (m²)": k * 400, "ships per pass (median)": int(np.median(n)),
                     "ships per day": round(mid), "90 % interval": f"{lo:.0f}-{hi:.0f}"})
    return pd.DataFrame(rows)


def plot_passes(df):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 3.8))
    ax.bar(pd.to_datetime(df.date), df.ships, width=6, color="#1f5fd6")
    ax.axhline(df.ships.mean(), ls="--", color="k", lw=1)
    ax.set_title("Ships in the gate at each Sentinel-1 pass, 2025 (19:34 local time, every 12 days)", loc="left", fontweight="bold")
    ax.set_ylabel("ships detected"); ax.grid(axis="y", alpha=0.3)
    return fig


def plot_lanes():
    import matplotlib.pyplot as plt
    m, _ = ships(3)
    fig, ax = plt.subplots(figsize=(10, 4.2))
    hb = ax.hexbin(m.lon, m.lat, gridsize=(60, 20), extent=GATE[0:1] + GATE[2:3] + GATE[1:2] + GATE[3:4],
                   bins="log", cmap="magma", mincnt=1)
    fig.colorbar(hb, ax=ax, label="detections in 2025")
    ax.set_aspect(1 / np.cos(np.radians(2.9)))
    ax.set_title("Where the ships were, 2025: the lanes appear from detections alone", loc="left", fontweight="bold")
    ax.set_xlabel("Longitude (°E)"); ax.set_ylabel("Latitude (°N)"); ax.grid(alpha=0.3)
    return fig


# Earth Engine: the radar itself --------------------------------------------------------------------
def s1(start, end, geom, pol, orbit=None):
    c = (ee.ImageCollection("COPERNICUS/S1_GRD").filterBounds(geom).filterDate(start, end)
         .filter(ee.Filter.eq("instrumentMode", "IW")).filter(ee.Filter.listContains("transmitterReceiverPolarisation", pol)))
    if orbit:
        c = c.filter(ee.Filter.eq("orbitProperties_pass", orbit))
    return c.select(pol).mosaic()


def products():
    bay = ee.Geometry.Rectangle(BAY, None, False)
    gate = ee.Geometry.Rectangle(GATE, None, False)
    vv = s1("2018-04-01T21:00", "2018-04-01T23:00", bay, "VV")
    vh = s1("2025-01-09", "2025-01-10", gate, "VH", "ASCENDING")
    return [
        {"kind": "map", "name": "ch77-balikpapan-vv", "image": vv.focalMean(50, "circle", "meters"),
         "vis": {"min": -28, "max": -10, "palette": ["000000", "ffffff"]}, "region": bay,
         "title": "Sentinel-1 VV, Balikpapan Bay, 2 April 2018 05:50 local (dB)", "source": "Copernicus Sentinel-1 GRD. GEE.",
         "caption": "Two days after the pipeline broke. The dark ribbon leaving the bay mouth and running east along the coast "
                    "is sea whose small waves the oil has calmed. Bright dots are ships at the anchorage."},
        {"kind": "figure", "name": "ch77-slick-detected", "figure": plot_slick,
         "caption": "Dark-spot detection: more than 3 dB darker than the sea within ~10 km, and darker than -24 dB. Orange: "
                    "large, high-contrast patches, likely oil. Blue: patches too small or too weak to call."},
        {"kind": "table", "name": "ch77-slick-patches", "data": slick_table,
         "caption": "Every dark patch over 0.5 km², with the features that separate oil from look-alikes."},
        {"kind": "map", "name": "ch77-malacca-vh", "image": vh,
         "vis": {"min": -28, "max": -8, "palette": ["000000", "ffffff"]}, "region": gate,
         "title": "Sentinel-1 VH over the gate, 9 January 2025 19:34 local (dB)", "source": "Copernicus Sentinel-1 GRD. GEE.",
         "caption": "One pass over the gate across the strait. Ships are the bright points; the sea is dark and grainy."},
        {"kind": "chart", "name": "ch77-ships-per-pass", "data": per_pass, "plot": plot_passes,
         "caption": "Ships counted at each of the 29 passes of 2025, after removing fixed targets. Dashed: the mean."},
        {"kind": "figure", "name": "ch77-lanes", "figure": plot_lanes,
         "caption": "All 2025 detections binned. Traffic concentrates in lanes; the sparse middle is shallow water and banks."},
        {"kind": "table", "name": "ch77-flow", "data": flow_table,
         "caption": "Ships per day through the gate, for five definitions of \"ship\". STRAITREP counts 281 per day "
                    "(vessels over 300 GT)."},
    ]


if __name__ == "__main__":
    print(slick_table().head())
    print(flow_table())
