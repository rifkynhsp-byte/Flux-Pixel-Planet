#| title: LiDAR, and passive against active (Python)
#| description: Passive and active sensing drawn side by side, a full-waveform return built from a canopy profile with its relative-height metrics, and real GEDI canopy height over a Kalimantan peat forest and its plantations.

# Fullstack Remote Sensing | Code by Rifky Nauval Hendrawan, Remote Sensing Specialist
# MIT licence: free to use and adapt; please keep this credit line.

"""
PRINCIPLES P8 | LiDAR: timing light to measure height.

    range  = c t / 2          (t: round-trip time; 1 ns of timing = 15 cm of range)
    height = range to ground - range to the top of the canopy

A passive sensor records light it did not send (sunlight, the Earth's own
heat). An active sensor sends its own pulse and times the echo: radar with
microwaves (P5), LiDAR with a laser.
"""

import ee
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle

C = 2.99792458e8


# ---------------------------------------------------------------------------
# 1. Passive against active, drawn
# ---------------------------------------------------------------------------
def passive_active_figure():
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, title in zip(axes, ["Passive: records energy it did not send",
                                "Active: sends a pulse and times the echo"]):
        ax.set_xlim(0, 10); ax.set_ylim(0, 7); ax.set_axis_off(); ax.set_aspect("equal")
        ax.add_patch(Rectangle((0, 0), 10, 0.8, color="#d9f99d"))
        for x in (3.2, 4.4, 5.6):                                   # trees
            ax.add_patch(Rectangle((x - 0.07, 0.8), 0.14, 0.9, color="#78350f"))
            ax.add_patch(Circle((x, 2.0), 0.55, color="#15803d"))
        ax.add_patch(Rectangle((6.6, 5.6), 1.1, 0.6, color="#334155"))   # satellite
        ax.text(7.15, 6.45, "sensor", ha="center", fontsize=8)
        ax.set_title(title, loc="left", fontsize=9.5)
    a1, a2 = axes
    a1.add_patch(Circle((1.0, 6.0), 0.5, color="#facc15")); a1.text(1.0, 5.2, "Sun", ha="center", fontsize=8)
    a1.add_patch(FancyArrowPatch((1.4, 5.6), (4.2, 2.6), arrowstyle="-|>", mutation_scale=12, color="#f59e0b", lw=1.8))
    a1.add_patch(FancyArrowPatch((4.6, 2.6), (6.9, 5.5), arrowstyle="-|>", mutation_scale=12, color="#f59e0b", lw=1.8))
    a1.add_patch(FancyArrowPatch((5.9, 1.2), (7.4, 5.5), arrowstyle="-|>", mutation_scale=12, color="#dc2626", lw=1.4, ls="--"))
    a1.text(0.2, 0.2, "Reflected sunlight: optical (P1, P3)   Emitted heat: thermal, passive microwave",
            fontsize=7, color="#1f2937")
    for x0, col, lab in [(6.9, "#2563eb", "laser (LiDAR)"), (7.5, "#7c3aed", "microwave (radar)")]:
        a2.add_patch(FancyArrowPatch((x0, 5.5), (x0 - 2.2, 2.4), arrowstyle="-|>", mutation_scale=12, color=col, lw=1.8))
        a2.add_patch(FancyArrowPatch((x0 - 2.0, 2.6), (x0 + 0.2, 5.5), arrowstyle="-|>", mutation_scale=12, color=col, lw=1.2, ls="--"))
    a2.text(0.2, 0.2, "Range = c × time / 2. Works at night; radar also through cloud, LiDAR not",
            fontsize=7, color="#1f2937")
    a2.text(8.2, 4.2, "laser\n(LiDAR)", fontsize=7.5, color="#2563eb")
    a2.text(8.2, 3.2, "microwave\n(radar, P5)", fontsize=7.5, color="#7c3aed")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# 2. A full waveform from a canopy, and its relative heights
# ---------------------------------------------------------------------------
def waveform_table():
    """A tropical forest canopy 32 m tall, gap fraction set so ground still returns."""
    z = np.arange(-5, 40, 0.15)
    canopy = (np.exp(-0.5 * ((z - 24) / 4.5) ** 2) * 1.0 + np.exp(-0.5 * ((z - 12) / 3.5) ** 2) * 0.45)
    canopy[(z < 1) | (z > 32)] = 0
    ground = 0.55 * np.exp(-0.5 * (z / 0.25) ** 2)
    profile = canopy / canopy.sum() * 0.75 + ground / ground.sum() * 0.25
    sigma = 15e-9 * C / 2 / 2.355                      # 15 ns pulse (FWHM) in metres of height
    kernel = np.exp(-0.5 * (np.arange(-30, 31) * 0.15 / sigma) ** 2)
    wf = np.convolve(profile, kernel / kernel.sum(), mode="same")
    noise = np.random.default_rng(1).normal(0, 0.00008, wf.size)
    return pd.DataFrame({"height_m": z, "energy": wf + noise, "profile": profile})


def plot_waveform(df):
    z, e = df.height_m.to_numpy(), df.energy.to_numpy()
    sig = np.clip(e - 0.0003, 0, None)                # above the noise
    cum = np.cumsum(sig[::-1])[::-1] / sig.sum()      # energy from the top down
    gnd = z[np.argmax(np.where(z < 3, e, 0))]
    rh = {p: z[np.argmin(np.abs(cum - (1 - p / 100)))] - gnd for p in (25, 50, 75, 98)}
    fig, ax = plt.subplots(figsize=(5.6, 6))
    ax.fill_betweenx(z - gnd, 0, e, color="#93c5fd")
    ax.plot(e, z - gnd, color="#1d4ed8", lw=1.4)
    for p, h in rh.items():
        ax.axhline(h, color="#475569", lw=0.8, ls=":")
        ax.text(e.max() * 1.05, h + 0.4, f"RH{p} = {h:.1f} m", va="bottom", fontsize=8)
    ax.text(e.max() * 0.35, -1.8, "ground peak", fontsize=8, color="#1d4ed8")
    ax.set_xlim(0, e.max() * 1.45); ax.set_ylim(-4, 38)
    ax.set_xlabel("Returned energy")
    ax.text(0, 1.02, "Height above ground (m)", transform=ax.transAxes, fontsize=8.5, color="#475569")
    ax.set_title(f"One footprint: canopy top RH98 = {rh[98]:.0f} m,\nhalf the energy above {rh[50]:.0f} m",
                 loc="left", fontsize=10, pad=18)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    return fig


# ---------------------------------------------------------------------------
# 3. GEDI canopy height around Palangka Raya and the edge of Sebangau, Central Kalimantan
# ---------------------------------------------------------------------------
kalteng = ee.Geometry.Rectangle([113.70, -2.45, 114.40, -1.95], None, False)   # Palangka Raya and Sebangau's edge


def good(img):
    return (img.updateMask(img.select("quality_flag").eq(1))
               .updateMask(img.select("degrade_flag").eq(0)))


gedi = (ee.ImageCollection("LARSE/GEDI/GEDI02_A_002_MONTHLY").filterBounds(kalteng)
        .filterDate("2019-04-01", "2024-01-01").map(good).select("rh98").max().clip(kalteng))
dw = (ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1").filterBounds(kalteng)
      .filterDate("2022-01-01", "2024-01-01").select("label").mode())


def height_samples():
    s = gedi.rename("rh98").addBands(dw.rename("cls")).stratifiedSample(
        numPoints=400, classBand="cls", region=kalteng, scale=25, seed=1, geometries=False)
    df = pd.DataFrame([f["properties"] for f in s.getInfo()["features"]])
    names = {1: "Trees", 5: "Shrub and scrub", 4: "Crops", 6: "Built", 7: "Bare"}
    df = df[df.cls.isin(names)]
    df["class"] = df.cls.map(names)
    return df[["class", "rh98"]]


def plot_heights(df):
    fig, ax = plt.subplots(figsize=(8, 3.8))
    order = df.groupby("class").rh98.median().sort_values().index      # tallest at the top
    for i, c in enumerate(order):
        v = df[df["class"] == c].rh98
        ax.boxplot(v, positions=[i], widths=0.55, vert=False, showfliers=False,
                   medianprops=dict(color="#dc2626"))
        q1, q3 = v.quantile([0.25, 0.75])
        ax.text(min(q3 + 1.5 * (q3 - q1), v.max()) + 0.8, i, f"median {v.median():.0f} m, n = {len(v)}",
                va="center", fontsize=7.5)
    ax.set_yticks(range(len(order))); ax.set_yticklabels(order, fontsize=8.5)
    ax.set_xlabel("GEDI RH98 (m)")
    top = df[df["class"] == "Trees"].rh98.median()
    ax.set_title(f"GEDI separates forest (median {top:.0f} m) from the cleared and burned land around it",
                 loc="left", fontsize=10)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    return fig


def lidar_table():
    return pd.DataFrame([
        ["Airborne (ALS)", "aircraft, 500-3000 m", "1064 or 1550", "a few to tens of points per m²",
         "one survey area at a time", "terrain, canopy, cities; about 10-15 cm vertical"],
        ["Drone (ULS)", "drone, 50-120 m", "905 or 1550", "hundreds of points per m²", "a few km²",
         "single trees, stockpiles, corridors"],
        ["GEDI", "ISS, about 420 km", "1064", "25 m footprints along 8 tracks", "51.6° N to 51.6° S, 2019-",
         "canopy height and structure (paused March 2023 to April 2024)"],
        ["ICESat-2 (ATLAS)", "satellite, about 500 km", "532", "photon counting, shots 0.7 m apart", "global, 2018-",
         "ice, sea ice, water level, terrain and canopy"],
    ], columns=["System", "Platform", "Wavelength (nm)", "Sampling", "Coverage", "Best for"])


HEIGHTS = height_samples()


def products():
    return [
        {"kind": "figure", "name": "p8-passive-active", "figure": passive_active_figure,
         "caption": "Passive sensors record sunlight reflected by the ground, or heat the ground "
                    "emits. Active sensors send their own pulse and time the echo: a laser for "
                    "LiDAR, microwaves for radar."},
        {"kind": "chart", "name": "p8-waveform", "data": waveform_table(), "plot": plot_waveform,
         "caption": "A simulated full waveform for one footprint over tropical forest. Energy "
                    "returns first from the top of the canopy, then from the lower layers, and "
                    "last from the ground. RH98 is the height below which 98 % of the energy "
                    "returned: the usual measure of canopy height."},
        {"kind": "map", "name": "p8-gedi-rh98", "image": gedi, "region": kalteng,
         "vis": {"min": 0, "max": 40, "palette": ["#fef9c3", "#a3e635", "#16a34a", "#14532d"]},
         "legend": "GEDI RH98, tallest of 2019-2023 (m)",
         "title": "Palangka Raya and Sebangau, GEDI canopy height", "source": "GEDI L2A monthly rasters. GEE.",
         "caption": "Every coloured dot is a GEDI footprint track: a laser sample, not a map. "
                    "Tall forest at the edge of the Sebangau peat swamp stands out against the "
                    "city, the cleared and burned peat and the farmland around Palangka Raya."},
        {"kind": "chart", "name": "p8-heights", "data": HEIGHTS, "plot": plot_heights,
         "caption": "GEDI RH98 of good-quality footprints, grouped by the Dynamic World class of "
                    "2022-2023 under each."},
        {"kind": "table", "name": "p8-systems", "data": lidar_table,
         "caption": "LiDAR systems from drone to orbit. Figures are typical, not specifications."},
    ]


if __name__ == "__main__":
    ee.Initialize()
    print(HEIGHTS.groupby("class").rh98.describe())
