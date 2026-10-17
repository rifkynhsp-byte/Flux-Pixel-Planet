#| title: Hyperspectral against multispectral (Python)
#| description: Real EMIT reflectance spectra over Bandung and Saguling, averaged by land cover, then squeezed into Sentinel-2 and Landsat bands to show what a multispectral sensor keeps and what it throws away.

# Fullstack Remote Sensing | Code by Rifky Nauval Hendrawan, Remote Sensing Specialist
# MIT licence: free to use and adapt; please keep this credit line.

"""
PRINCIPLES P6 | Hyperspectral against multispectral.

    multispectral   a few wide bands placed in atmospheric windows
                    (Sentinel-2: 13 bands, 15 to 180 nm wide)
    hyperspectral   hundreds of narrow, contiguous bands
                    (EMIT: 285 bands, about 7.4 nm apart, 381 to 2493 nm)

A multispectral band is roughly the average of the hyperspectral bands that
fall inside it. Everything that happens within that average, a shoulder, a
dip, a shift, is lost.
"""

import io
import math
from pathlib import Path

import ee
import matplotlib.pyplot as plt
import requests
from matplotlib.patches import Rectangle
from PIL import Image
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# 1. EMIT surface reflectance: the clearest dry-season pass, 26 August 2025
# ---------------------------------------------------------------------------
aoi = ee.Geometry.Rectangle([107.30, -7.05, 107.75, -6.75], None, False)   # Saguling to Bandung
emit = ee.ImageCollection("NASA/EMIT/L2A/RFL").filterBounds(aoi)
WAVE = ee.List(emit.first().get("reflectance_wavelengths"))
RBANDS = [f"reflectance_{i}" for i in range(285)]


def clear(img):
    bad = (img.select("cloud_flag").Or(img.select("cirrus_flag"))
           .Or(img.select("dilated_cloud_flag")))
    refl = img.select(RBANDS)
    return refl.updateMask(bad.Not()).updateMask(refl.select("reflectance_60").gt(0))


SCENE = "EMIT_L2A_RFL_001_20250826T041241_2523802_036"   # 75 % of the box clear
rfl = clear(emit.filter(ee.Filter.eq("system:index", SCENE)).first()).clip(aoi)

# Land cover from Dynamic World, most common label of the same years
dw = (ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1").filterBounds(aoi)
      .filterDate("2025-06-01", "2025-10-01").select("label").mode())
CLASSES = {0: "Water", 1: "Trees", 4: "Crops", 6: "Built"}
COLOURS = {"Water": "#1f78b4", "Trees": "#1b7837", "Crops": "#d9a400", "Built": "#7f7f7f"}


def class_spectra():
    """Mean EMIT spectrum of each land-cover class, one row per wavelength.

    Only pixels at least 90 m inside a class are used, so that a 60 m EMIT
    pixel is not half water and half shore.
    """
    wl = np.array(WAVE.getInfo())
    rows = {"wavelength_nm": wl}
    for code, name in CLASSES.items():
        core = dw.eq(code).focalMin(90, "circle", "meters")   # away from edges: pure pixels
        m = rfl.updateMask(core).reduceRegion(
            ee.Reducer.mean(), aoi, scale=60, maxPixels=1e9, tileScale=4).getInfo()
        rows[name] = [m.get(b) for b in RBANDS]
    df = pd.DataFrame(rows)
    # Deep water-vapour absorption: the atmosphere leaves nothing to measure
    gap = df.wavelength_nm.between(1320, 1460) | df.wavelength_nm.between(1770, 1970) \
        | (df.wavelength_nm > 2450)
    df.loc[gap, list(CLASSES.values())] = np.nan
    return df


# Band windows (centre, width in nm), simplified to rectangles
S2 = {"B1": (443, 20), "B2": (490, 65), "B3": (560, 35), "B4": (665, 30), "B5": (705, 15),
      "B6": (740, 15), "B7": (783, 20), "B8": (842, 115), "B8A": (865, 20), "B9": (945, 20),
      "B11": (1610, 90), "B12": (2190, 180)}
L8 = {"B1": (443, 16), "B2": (482, 60), "B3": (561, 57), "B4": (655, 37), "B5": (865, 28),
      "B6": (1609, 85), "B7": (2201, 187)}


def resample(df, bands):
    """What a multispectral sensor would report: the mean inside each band."""
    out = []
    for name, (c, w) in bands.items():
        inside = df[df.wavelength_nm.between(c - w / 2, c + w / 2)]
        row = {"band": name, "centre_nm": c, "width_nm": w}
        for cls in CLASSES.values():
            row[cls] = inside[cls].mean()
        out.append(row)
    return pd.DataFrame(out)


def plot_spectra(df):
    s2 = resample(df, S2)
    fig, ax = plt.subplots(figsize=(10, 4.6))
    for cls, col in COLOURS.items():
        ax.plot(df.wavelength_nm, df[cls], color=col, lw=1.6, label=cls)
        ax.plot(s2.centre_nm, s2[cls], "o", color=col, ms=6, mec="white", mew=0.8)
    for _, b in s2.iterrows():
        ax.axvspan(b.centre_nm - b.width_nm / 2, b.centre_nm + b.width_nm / 2,
                   color="#94a3b8", alpha=0.12, lw=0)
    ax.text(0.99, 0.95, "lines: EMIT, 285 bands; dots: the same spectra in Sentinel-2's 12 bands (grey)",
            transform=ax.transAxes, ha="right", fontsize=8, color="#475569")
    ax.set_xlabel("Wavelength (nm)")
    ax.text(0, 1.03, "Surface reflectance", transform=ax.transAxes, fontsize=9, color="#475569")
    ax.set_xlim(380, 2460)
    ax.set_ylim(0, None)
    ax.legend(frameon=False, fontsize=8, loc="upper right", bbox_to_anchor=(1.0, 0.9))
    ax.set_title("285 bands draw the whole curve; Sentinel-2's 12 keep its outline and lose the detail",
                 loc="left", fontsize=11, pad=22)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    return fig


def red_edge_position(wl, refl):
    """Wavelength of the steepest rise between 680 and 750 nm."""
    sel = (wl >= 680) & (wl <= 750)
    d = np.gradient(refl[sel], wl[sel])
    return float(wl[sel][np.nanargmax(d)])


def plot_detail(df):
    wl = df.wavelength_nm.to_numpy()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 4.2))
    # (a) red edge: trees against crops
    for cls in ("Trees", "Crops"):
        r = df[cls].to_numpy()
        sel = (wl >= 650) & (wl <= 800)
        a1.plot(wl[sel], r[sel], "-o", ms=2.5, color=COLOURS[cls], lw=1.4,
                label=f"{cls}: red edge at {red_edge_position(wl, r):.0f} nm")
    for b in ("B4", "B5", "B6", "B7"):
        c, w = S2[b]
        a1.axvspan(c - w / 2, c + w / 2, color="#94a3b8", alpha=0.18, lw=0)
        a1.text(c, 0.98, b, ha="center", va="top", fontsize=7, color="#475569",
                transform=a1.get_xaxis_transform())
    a1.set_xlabel("Wavelength (nm)")
    a1.legend(frameon=False, fontsize=8, loc="lower right")
    a1.set_title("a. Red edge: 20 EMIT bands, 4 Sentinel-2 bands", loc="left", fontsize=9.5)
    # (b) shortwave infrared beyond 2000 nm: one wide Sentinel-2 band
    for cls in ("Built", "Crops", "Trees"):
        r = df[cls].to_numpy()
        sel = (wl >= 2000) & (wl <= 2440)
        a2.plot(wl[sel], r[sel], "-", color=COLOURS[cls], lw=1.6, label=cls)
    c, w = S2["B12"]
    a2.axvspan(c - w / 2, c + w / 2, color="#94a3b8", alpha=0.18, lw=0)
    a2.text(c, 0.02, "Sentinel-2 B12: one number", ha="center", transform=a2.get_xaxis_transform(),
            fontsize=7.5, color="#475569")
    a2.set_xlabel("Wavelength (nm)")
    a2.legend(frameon=False, fontsize=8)
    a2.set_title("b. 2000 to 2440 nm: about 60 EMIT bands, 1 Sentinel-2 band", loc="left",
                 fontsize=9.5)
    for ax in (a1, a2):
        ax.set_ylabel("Surface reflectance", fontsize=8, color="#475569")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.tight_layout()
    return fig


def band_table(df):
    s2 = resample(df, S2)
    s2.insert(0, "sensor", "Sentinel-2")
    return s2.round(3)


def sensors_table():
    return pd.DataFrame([
        ["EMIT (ISS)", "hyperspectral", 285, "381-2493", "about 7.4", 60, "2022-"],
        ["EnMAP", "hyperspectral", "about 250", "420-2450", "6.5 / 10", 30, "2022-"],
        ["PRISMA", "hyperspectral", 239, "400-2500", "about 12", 30, "2019-"],
        ["Hyperion (EO-1)", "hyperspectral", "242 (198 usable)", "357-2576", "about 10", 30, "2000-2017"],
        ["Sentinel-2 MSI", "multispectral", 13, "443-2190", "15-180", "10/20/60", "2015-"],
        ["Landsat 8/9 OLI", "multispectral", "9", "433-2294", "16-187", "30 (15 pan)", "2013-"],
    ], columns=["Sensor", "Type", "Bands", "Range (nm)", "Band width (nm)", "Pixel (m)", "Years"])


# ---------------------------------------------------------------------------
# 3. From the street to the pixel: four places seen from the ground and from orbit
# ---------------------------------------------------------------------------
# Street View panoramas found near each place; the camera looks at a point a
# few tens of metres away whose Dynamic World class matches the place.
SITES = [
    {"name": "Building, Gedung Merdeka, Bandung", "pano": (-6.92133, 107.60981), "heading": 300,
     "dist": 25, "sv_date": "May 2024"},
    {"name": "Paddy, Majalaya", "pano": (-7.03767, 107.74397), "heading": 180, "dist": 45,
     "sv_date": "April 2021"},
    {"name": "Lake, Situ Cileunca", "pano": (-7.19754, 107.54934), "heading": 320, "dist": 80,
     "sv_date": "December 2021"},
    {"name": "Forest, Tahura Juanda", "pano": (-6.85715, 107.63307), "heading": 60, "dist": 30,
     "sv_date": "November 2019"},
]
SV_DIR = Path(__file__).resolve().parents[2] / "images" / "real"
DW_NAMES = ["water", "trees", "grass", "flooded vegetation", "crops", "shrub and scrub", "built",
            "bare", "snow and ice"]
IJB_NAMES = ["Plantation", "Forest", "Agriculture", "Open land", "Water", "Mangrove", "Paddy",
             "Oil palm", "Sago", "Rubber"]


def target_point(site):
    lat, lon = site["pano"]
    b = math.radians(site["heading"])
    dlat = site["dist"] * math.cos(b) / 6371000
    dlon = site["dist"] * math.sin(b) / (6371000 * math.cos(math.radians(lat)))
    return lat + math.degrees(dlat), lon + math.degrees(dlon)


def street_view(site, i):
    """The Street View photo, fetched once with the author's key and kept with attribution."""
    out = SV_DIR / f"p6-streetview-{i}.jpg"
    key_file = Path.home() / ".config" / "qsgi" / "maps-api-key"
    if not out.exists() and key_file.exists():
        r = requests.get("https://maps.googleapis.com/maps/api/streetview",
                         params={"size": "640x400", "location": "%f,%f" % site["pano"],
                                 "heading": site["heading"], "fov": 70, "pitch": 0,
                                 "source": "outdoor", "key": key_file.read_text().strip()},
                         timeout=60)
        if r.headers.get("content-type", "").startswith("image/"):
            out.write_bytes(r.content)
    return Image.open(out) if out.exists() else None


S2B = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]
S2C = [490, 560, 665, 705, 740, 783, 842, 865, 1610, 2190]


def ground_truth_table():
    """For each target pixel: Sentinel-2 reflectance, indices, Dynamic World and the 10-class model."""
    pts = ee.FeatureCollection([ee.Feature(ee.Geometry.Point(target_point(st)[::-1]), {"site": st["name"]})
                                for st in SITES])
    box = pts.geometry().buffer(2000).bounds()
    s2 = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").filterBounds(box)
          .filterDate("2025-06-01", "2025-10-01").filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 40))
          .map(lambda im: im.updateMask(im.select("SCL").remap([4, 5, 6, 11], [1, 1, 1, 1], 0)))
          .select(S2B).median().divide(10000))
    ndvi = s2.normalizedDifference(["B8", "B4"]).rename("NDVI")
    evi = s2.expression("2.5*(N-R)/(N+6*R-7.5*B+1)", {"N": s2.select("B8"), "R": s2.select("B4"),
                                                       "B": s2.select("B2")}).rename("EVI")
    dwc = (ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1").filterBounds(box).filterDate("2025-06-01", "2025-10-01"))
    dw_label = dwc.select("label").mode().rename("dw")
    dw_prob = dwc.select(["water", "trees", "grass", "flooded_vegetation", "crops",
                                           "shrub_and_scrub", "built", "bare"]).mean()
    emb = (ee.ImageCollection("GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL")
           .filter(ee.Filter.calendarRange(2024, 2024, "year")).filterBounds(box).mosaic())
    dem = ee.Image("USGS/SRTMGL1_003").select("elevation")
    ijb = (ee.Image.cat([emb, dem, ee.Terrain.slope(dem)])
           .classify(ee.Classifier.load("projects/ee-rifkynauvalhsp2/assets/LULC_Classifier_Embeddings_Topo_rubber"))
           .rename("ijb"))
    stack = s2.addBands([ndvi, evi, dw_label, ijb]).addBands(dw_prob.max(ee.Image(0)).reduce(ee.Reducer.max()).rename("dw_p"))
    rows = stack.sampleRegions(pts, scale=10).getInfo()["features"]
    df = pd.DataFrame([r["properties"] for r in rows])
    df["dw"] = df.dw.astype(int).map(lambda v: DW_NAMES[v])
    df["ijb"] = df.ijb.astype(int).map(lambda v: IJB_NAMES[v])
    return df.set_index("site").loc[[st["name"] for st in SITES]].reset_index()


def s2_chip(site):
    lat, lon = target_point(site)
    d = 0.0018                                         # about 200 m either side
    region = ee.Geometry.Rectangle([lon - d, lat - d, lon + d, lat + d], None, False)
    img = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").filterBounds(region)
           .filterDate("2025-06-01", "2025-10-01").filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 40))
           .select(["B4", "B3", "B2"]).median())
    url = img.visualize(min=200, max=2000, gamma=1.2).getThumbURL(
        {"region": region, "dimensions": 400, "format": "png", "crs": "EPSG:4326"})
    return Image.open(io.BytesIO(requests.get(url, timeout=300).content)), (lon - d, lon + d, lat - d, lat + d)


def plot_street_to_pixel(df):
    n = len(SITES)
    fig, axes = plt.subplots(n, 3, figsize=(13, 3.3 * n), gridspec_kw={"width_ratios": [1.6, 1, 1.5]})
    for i, site in enumerate(SITES):
        row = df.iloc[i]
        a0, a1, a2 = axes[i]
        sv = street_view(site, i)
        if sv is not None:
            a0.imshow(sv)
        a0.set_axis_off()
        a0.set_title(f"{site['name']}: Street View, {site['sv_date']}", loc="left", fontsize=9)
        chip, ext = s2_chip(site)
        a1.imshow(chip, extent=ext)
        tl = target_point(site)
        a1.plot(site["pano"][1], site["pano"][0], "^", color="#facc15", ms=8, mec="black", label="camera")
        a1.add_patch(Rectangle((tl[1] - 0.000045, tl[0] - 0.000045), 0.00009, 0.00009, fill=False,
                               ec="#ef4444", lw=1.8))
        a1.set_xticks([]); a1.set_yticks([])
        a1.set_title("Sentinel-2, 2025: camera ▲, pixel looked at □", loc="left", fontsize=8)
        refl = [row[b] for b in S2B]
        a2.plot(S2C, refl, "-o", color="#1d4ed8", ms=4)
        a2.set_ylim(0, max(0.45, max(refl) + 0.04)); a2.set_xlim(400, 2300)
        a2.set_title(f"Dynamic World: {row.dw}   |   10-class model: {row.ijb}", loc="left", fontsize=8.5)
        a2.text(0.98, 0.95, f"NDVI {row.NDVI:.2f}\nEVI {row.EVI:.2f}", transform=a2.transAxes,
                ha="right", va="top", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#cbd5e1"))
        a2.set_ylabel("Reflectance", fontsize=8)
        if i == n - 1:
            a2.set_xlabel("Wavelength (nm)")
        for sp in ("top", "right"):
            a2.spines[sp].set_visible(False)
    fig.suptitle("What you see from the street, what the satellite sees, and what two classifiers call it",
                 x=0.01, ha="left", fontsize=11.5)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig


# ---------------------------------------------------------------------------
# 4. A cross-section through Kota Baru Parahyangan: water, hills, farms, town
# ---------------------------------------------------------------------------
XA, XB = (107.440, -6.905), (107.512, -6.833)      # Saguling shore (SW) to Padalarang (NE)
XN = 180                                             # samples, about 60 m apart
DW_COL = ["#419bdf", "#397d49", "#88b053", "#7a87c6", "#e49635", "#dfc35a", "#c4281b", "#a59b8f", "#b39fe1"]
NORM = {"NDVI": (-0.2, 0.9), "EVI": (-0.1, 0.8), "VV": (-20, 0), "VH": (-27, -8), "CHM": (0, 35)}
STYLE = {"NDVI": dict(color="#16a34a", ls="-", marker="o"), "EVI": dict(color="#ca8a04", ls="--", marker="s"),
         "VV": dict(color="#7c3aed", ls=":", marker="^"), "VH": dict(color="#db2777", ls="-.", marker="v"),
         "CHM": dict(color="#0f766e", ls="-", marker="D")}
PHOTO = [(0, "water"), (4, "crops"), (1, "forest"), (6, "built-up")]     # Dynamic World code, label


def transect_table():
    xs = np.linspace(XA[0], XB[0], XN)
    ys = np.linspace(XA[1], XB[1], XN)
    pts = ee.FeatureCollection([ee.Feature(ee.Geometry.Point(float(x), float(y)), {"i": i})
                                for i, (x, y) in enumerate(zip(xs, ys))])
    box = pts.geometry().buffer(1000).bounds()
    s2 = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").filterBounds(box)
          .filterDate("2025-06-01", "2025-10-01").filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 40))
          .map(lambda im: im.updateMask(im.select("SCL").remap([4, 5, 6, 11], [1, 1, 1, 1], 0)))
          .median().divide(10000))
    ndvi = s2.normalizedDifference(["B8", "B4"]).rename("NDVI")
    evi = s2.expression("2.5*(N-R)/(N+6*R-7.5*B+1)", {"N": s2.select("B8"), "R": s2.select("B4"),
                                                       "B": s2.select("B2")}).rename("EVI")
    s1 = (ee.ImageCollection("COPERNICUS/S1_GRD").filterBounds(box).filterDate("2025-06-01", "2025-10-01")
          .filter(ee.Filter.eq("instrumentMode", "IW")).select(["VV", "VH"]).median())
    chm = ee.Image("users/nlang/ETH_GlobalCanopyHeight_2020_10m_v1").select(0).rename("CHM")
    dem = ee.ImageCollection("COPERNICUS/DEM/GLO30").select("DEM").mosaic().rename("elev")
    dw = (ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1").filterBounds(box)
          .filterDate("2025-06-01", "2025-10-01").select("label").mode().rename("dw"))
    rows = (ee.Image.cat([ndvi, evi, s1, chm, dem, dw]).reduceRegions(pts, ee.Reducer.first(), 10)
            .getInfo()["features"])
    df = pd.DataFrame([r["properties"] for r in rows]).sort_values("i").reset_index(drop=True)
    df["lon"], df["lat"] = xs[df.i], ys[df.i]
    df["km"] = np.hypot((df.lon - XA[0]) * 111.32 * math.cos(math.radians(XA[1])), (df.lat - XA[1]) * 110.57)
    return df


# Street View photos chosen by eye from Google's own imagery near the line, no people
# in view: (label, pano lat, pano lon, heading, target lon, target lat, date)
PHOTOS = [("water", -6.90224, 107.44521, 230, 107.44466, -6.90324, "June 2024"),
          ("crops", -6.90054, 107.44481, 266, 107.44443, -6.90057, "August 2021"),
          ("built-up", -6.87663, 107.46649, 188, 107.46617, -6.87883, "June 2024"),
          ("forest", -6.83972, 107.50981, 237, 107.50936, -6.84002, "July 2024")]


def photo_points(df):
    """The chosen photos, fetched once with the author's key and kept with Google's attribution."""
    key_file = Path.home() / ".config" / "qsgi" / "maps-api-key"
    out = []
    dx, dy = XB[0] - XA[0], XB[1] - XA[1]
    for label, plat, plon, head, tlon, tlat, date in PHOTOS:
        f = SV_DIR / f"p6-kbp-{label}.jpg"
        if not f.exists() and key_file.exists():
            r = requests.get("https://maps.googleapis.com/maps/api/streetview",
                             params={"size": "640x400", "location": f"{plat},{plon}", "heading": head,
                                     "pitch": -5, "fov": 70, "source": "outdoor",
                                     "key": key_file.read_text().strip()}, timeout=60)
            if r.headers.get("content-type", "").startswith("image/"):
                f.write_bytes(r.content)
        t = ((tlon - XA[0]) * dx + (tlat - XA[1]) * dy) / (dx * dx + dy * dy)
        row = df.iloc[(df.i - t * (XN - 1)).abs().argmin()]
        out.append(dict(label=label, row=row, file=f, date=date, plat=plat, plon=plon, tlon=tlon, tlat=tlat))
    return out


def line_coords(lon, lat):
    """Distance along the line from A, and across it (left positive), in km."""
    kx, ky = 111.32 * math.cos(math.radians(XA[1])), 110.57
    ex, ey = (XB[0] - XA[0]) * kx, (XB[1] - XA[1]) * ky
    L = math.hypot(ex, ey)
    px, py = (lon - XA[0]) * kx, (lat - XA[1]) * ky
    return (px * ex + py * ey) / L, (ex * py - ey * px) / L


def plot_transect(df):
    photos = sorted(photo_points(df), key=lambda ph: ph["row"].km)      # numbered A to B
    L = float(df.km.max())
    fig = plt.figure(figsize=(13, 9.2))
    gs = fig.add_gridspec(4, 4, height_ratios=[1.5, 1.1, 2.2, 0.28], hspace=0.3, wspace=0.06,
                          top=0.94)
    # 1. Photos in order along the line
    for n, ph in enumerate(photos):
        ax = fig.add_subplot(gs[0, n])
        if ph["file"].exists():
            ax.imshow(Image.open(ph["file"]))
        ax.set_axis_off()
        ax.set_title(f"{n + 1}. {ph['label']}, km {ph['row'].km:.1f}", loc="left", fontsize=9.5)
        ax.text(0.99, 0.03, ph["date"], transform=ax.transAxes, ha="right", fontsize=7, color="white")
    # 2. Sentinel-2 strip, 1 km either side, rotated so A to B runs left to right
    kx, ky = 111.32 * math.cos(math.radians(XA[1])), 110.57
    x0, x1 = min(XA[0], XB[0]) - 0.015, max(XA[0], XB[0]) + 0.015
    y0, y1 = min(XA[1], XB[1]) - 0.015, max(XA[1], XB[1]) + 0.015
    region = ee.Geometry.Rectangle([x0, y0, x1, y1], None, False)
    rgb = (ee.ImageCollection("COPERNICUS/S2_SR_HARMONIZED").filterBounds(region)
           .filterDate("2025-06-01", "2025-10-01").filter(ee.Filter.lt("CLOUDY_PIXEL_PERCENTAGE", 40))
           .select(["B4", "B3", "B2"]).median().visualize(min=200, max=2200, gamma=1.2))
    url = rgb.getThumbURL({"region": region, "dimensions": 1200, "format": "png", "crs": "EPSG:4326"})
    img = np.asarray(Image.open(io.BytesIO(requests.get(url, timeout=300).content)).convert("RGB"))
    H, W = img.shape[:2]
    ex, ey = (XB[0] - XA[0]) * kx / L, (XB[1] - XA[1]) * ky / L          # unit vector along, in km
    u, v = np.meshgrid(np.linspace(0, L, 900), np.linspace(1, -1, 200))
    lon = XA[0] + (u * ex - v * ey) / kx
    lat = XA[1] + (u * ey + v * ex) / ky
    col = np.clip(((lon - x0) / (x1 - x0) * (W - 1)).round().astype(int), 0, W - 1)
    row = np.clip(((y1 - lat) / (y1 - y0) * (H - 1)).round().astype(int), 0, H - 1)
    ar = fig.add_subplot(gs[1, :])
    ar.imshow(img[row, col], extent=[0, L, -1, 1], aspect="auto")
    ar.axhline(0, color="white", lw=1.2, ls="--")
    for n, ph in enumerate(photos, 1):
        cu, cv = line_coords(ph["plon"], ph["plat"])
        tu, tv = line_coords(ph["tlon"], ph["tlat"])
        ar.annotate("", xy=(tu, tv), xytext=(cu, cv), arrowprops=dict(arrowstyle="->", color="#facc15", lw=1.4))
        ar.plot(cu, cv, "^", color="#facc15", ms=9, mec="black")
        ar.text(cu, cv + 0.25, str(n), color="#facc15", fontsize=11, fontweight="bold", ha="center")
    ar.set_xlim(0, L); ar.set_ylim(-1, 1)
    ar.set_ylabel("km across", fontsize=8)
    ar.tick_params(labelsize=7, labelbottom=False)
    ar.set_title("Sentinel-2 true colour 2025, 1 km either side of the line A (Saguling shore) to "
                 "B (Padalarang); ▲ cameras 1-4", loc="left", fontsize=9)
    # 3. Profile: five measures with their own colour, line and marker; elevation behind
    ac = fig.add_subplot(gs[2, :], sharex=ar)
    ae = ac.twinx()
    ae.fill_between(df.km, df.elev.min() - 5, df.elev, color="#e2e8f0", zorder=0)
    ae.set_ylabel("Elevation (m), grey", fontsize=8, color="#64748b")
    ae.tick_params(labelsize=7, colors="#64748b")
    ac.set_zorder(ae.get_zorder() + 1); ac.patch.set_visible(False)
    for k, (lo, hi) in NORM.items():
        val = ((df[k] - lo) / (hi - lo)).clip(0, 1).rolling(3, center=True, min_periods=1).mean()
        st = STYLE[k]
        unit = " m" if k == "CHM" else (" dB" if k in ("VV", "VH") else "")
        ac.plot(df.km, val, color=st["color"], ls=st["ls"], marker=st["marker"], markevery=10, ms=5,
                lw=1.6, label=f"{k} ({lo} to {hi}{unit})")
    for n, ph in enumerate(photos, 1):
        ac.axvline(ph["row"].km, color="#94a3b8", lw=0.8)
        ac.text(ph["row"].km, 1.02, str(n), ha="center", fontsize=9, fontweight="bold", color="#b45309")
    ac.set_ylim(0, 1.08); ac.set_ylabel("Scaled 0 to 1", fontsize=8.5)
    ac.legend(frameon=False, fontsize=7.5, ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.2))
    ac.tick_params(labelbottom=False)
    # 4. Dynamic World strip
    ad = fig.add_subplot(gs[3, :], sharex=ar)
    for i in range(len(df) - 1):
        ad.axvspan(df.km.iloc[i], df.km.iloc[i + 1], color=DW_COL[int(df.dw.iloc[i])], lw=0)
    ad.set_yticks([])
    ad.set_xlabel("Distance from A (km); strip: Dynamic World 2025", fontsize=8.5)
    from matplotlib.patches import Patch
    seen = sorted(set(int(x) for x in df.dw))
    ad.legend(handles=[Patch(color=DW_COL[c], label=DW_NAMES[c]) for c in seen], frameon=False, fontsize=7.5,
              ncol=len(seen), loc="upper center", bbox_to_anchor=(0.5, -1.7))
    fig.suptitle("Kota Baru Parahyangan in one line: each sensor sees a different part of the landscape",
                 x=0.01, y=0.985, ha="left", fontsize=12)
    return fig

SPECTRA = class_spectra()
TRANSECT = transect_table()
GROUND = ground_truth_table()
RGB = {"bands": ["reflectance_35", "reflectance_23", "reflectance_11"],
       "min": 0, "max": 0.15, "gamma": 1.3}


def products():
    return [
        {"kind": "map", "name": "p6-emit-rgb", "image": rfl, "region": aoi, "vis": RGB,
         "legend": "EMIT true colour (641, 552, 463 nm)",
         "title": "Saguling to Bandung, EMIT, 26 Aug 2025", "source": "NASA EMIT L2A. GEE.",
         "caption": "EMIT reflectance in true colour: three of its 285 bands, from the clearest "
                    "dry-season pass over the area (26 August 2025, 60 m). Cloud is masked. "
                    "Saguling reservoir is on the left, Bandung on the right."},
        {"kind": "chart", "name": "p6-spectra", "data": SPECTRA, "plot": plot_spectra,
         "caption": "Mean EMIT spectrum of each Dynamic World class on 26 August 2025 (lines) and the same "
                    "spectra averaged into Sentinel-2 bands (dots). Gaps are the water-vapour "
                    "absorption bands, where too little light reaches the sensor."},
        {"kind": "chart", "name": "p6-detail", "data": SPECTRA, "plot": plot_detail,
         "caption": "Two places where the detail matters. (a) The red edge: its exact position "
                    "shifts with chlorophyll and canopy, and EMIT samples it every 7 nm. "
                    "(b) Beyond 2000 nm, where minerals, cellulose and moisture leave narrow "
                    "features, Sentinel-2 has one band 180 nm wide."},
        {"kind": "table", "name": "p6-s2-table", "data": SPECTRA, "transform": band_table,
         "caption": "The same class spectra as Sentinel-2 would report them: the mean "
                    "reflectance inside each band."},
        {"kind": "chart", "name": "p6-street-to-pixel", "data": GROUND, "plot": plot_street_to_pixel,
         "live": False,
         "caption": "Four places in and around Bandung. Left: Google Street View, looking at a point "
                    "a few tens of metres from the camera. Middle: the same place in Sentinel-2, "
                    "with the camera (yellow) and the 10 m pixel it looks at (red). Right: that "
                    "pixel's spectral signature in ten Sentinel-2 bands, its NDVI and EVI, and the "
                    "labels Dynamic World and the author's 10-class model give it. Street View "
                    "images © Google."},
        {"kind": "table", "name": "p6-street-table", "data": GROUND,
         "columns": ["site", "dw", "ijb", "NDVI", "EVI", "B4", "B8", "B11"],
         "floatfmt": ("", "", "", ".2f", ".2f", ".3f", ".3f", ".3f"),
         "caption": "The same four pixels as numbers: labels, indices and three key bands (red, "
                    "near infrared, shortwave infrared)."},
        {"kind": "chart", "name": "p6-transect", "data": TRANSECT, "plot": plot_transect,
         "caption": "One line, about 11 km, from the Saguling shore through the hills and farms "
                    "around Kota Baru Parahyangan to Padalarang, sampled every 60 m. Top: the line "
                    "on Sentinel-2. Middle: elevation, then five measures scaled to 0-1 over fixed "
                    "ranges (legend) with their own line styles, then Dynamic World. Bottom: Street "
                    "View at the middle of the longest stretch of each class (images © Google)."},
        {"kind": "table", "name": "p6-sensors", "data": sensors_table,
         "caption": "Hyperspectral and multispectral sensors. Band counts and ranges from the "
                    "mission documentation; widths rounded."},
    ]


if __name__ == "__main__":
    ee.Initialize()
    print(SPECTRA.describe())
