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

import ee
import matplotlib.pyplot as plt
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


SPECTRA = class_spectra()
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
        {"kind": "table", "name": "p6-sensors", "data": sensors_table,
         "caption": "Hyperspectral and multispectral sensors. Band counts and ranges from the "
                    "mission documentation; widths rounded."},
    ]


if __name__ == "__main__":
    ee.Initialize()
    print(SPECTRA.describe())
