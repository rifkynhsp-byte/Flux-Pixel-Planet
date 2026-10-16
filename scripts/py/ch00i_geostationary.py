#| title: Geostationary and navigation satellites (Python)
#| description: Which geostationary weather satellite sees each place, and how its pixels stretch towards the edge of the disk; where GNSS and weather satellites orbit; and the Jakarta New Year 2020 rain as GSMaP measured it hour by hour.

# Fullstack Remote Sensing | Code by Rifky Nauval Hendrawan, Remote Sensing Specialist
# MIT licence: free to use and adapt; please keep this credit line.

"""
PRINCIPLES P7 | Satellites that stay put, and satellites that tell you where you are.

    geostationary   35 786 km above the equator, one orbit per sidereal day:
                    the same view of a third of the Earth every 10 minutes
    GNSS            about 19 000 to 23 000 km, many satellites in several
                    planes: four or more always in view, anywhere

From geostationary height the Earth is a disk. A pixel that is 2 km wide
under the satellite is stretched towards the edge, by the distance and by
the slant of the view.
"""

import math
from pathlib import Path

import ee
import io

import h5py
import requests
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Circle

RE, H = 6378.137, 35786.0          # Earth's equatorial radius and geostationary height, km
RS = RE + H

GEO = {"Himawari-9 (Japan)": 140.7, "GK-2A (Korea)": 128.2, "Meteosat (Europe, 0°)": 0.0,
       "GOES-East (USA)": -75.2, "GOES-West (USA)": -137.2}


def view_geometry(lat, lon, sublon):
    """Satellite zenith angle (deg) and slant distance (km) for a ground point."""
    g = math.acos(math.cos(math.radians(lat)) * math.cos(math.radians(lon - sublon)))
    d = math.sqrt(RE ** 2 + RS ** 2 - 2 * RE * RS * math.cos(g))
    vza = math.degrees(math.asin(RS * math.sin(g) / d))
    return vza, d


def cities_table():
    cities = {"Jayapura": (-2.53, 140.72), "Makassar": (-5.15, 119.43), "Jakarta": (-6.18, 106.83),
              "Medan": (3.59, 98.67), "Banda Aceh": (5.55, 95.32), "Sydney": (-33.87, 151.21)}
    rows = []
    for name, (lat, lon) in cities.items():
        vza, d = view_geometry(lat, lon, GEO["Himawari-9 (Japan)"])
        stretch = (d / H) / math.cos(math.radians(vza))     # radial direction
        rows.append([name, round(vza, 1), round(d), round(2.0 * d / H, 1), round(2.0 * stretch, 1)])
    return pd.DataFrame(rows, columns=["Place", "Zenith angle from Himawari-9 (°)", "Distance (km)",
                                       "2 km IR pixel, across (km)", "2 km IR pixel, towards the edge (km)"])


# ---------------------------------------------------------------------------
# 1. Which geostationary satellite sees each place best (Earth Engine image)
# ---------------------------------------------------------------------------
ll = ee.Image.pixelLonLat()
lat, lon = ll.select("latitude").multiply(math.pi / 180), ll.select("longitude")


def vza_image(sublon):
    dlon = lon.subtract(sublon).multiply(math.pi / 180)
    cosg = lat.cos().multiply(dlon.cos())
    g = cosg.acos()
    d = ee.Image(RE ** 2 + RS ** 2).subtract(cosg.multiply(2 * RE * RS)).sqrt()
    vza = g.sin().multiply(RS).divide(d).asin().multiply(180 / math.pi)
    # Beyond the limb (cos g < RE/RS) the satellite cannot see the ground at all
    return vza.where(cosg.lt(RE / RS), 180)


names = list(GEO)
land = ee.Image().paint(ee.FeatureCollection("USDOS/LSIB_SIMPLE/2017"), 0, 1)
world = ee.Geometry.Rectangle([-179.9, -75, 179.9, 75], None, False)


def footprint(sublon):
    """What one satellite sees: viewing angle where usable (< 70°), with borders on top."""
    v = vza_image(sublon)
    return (v.updateMask(v.lt(70)).visualize(min=0, max=70, palette=["#1d4ed8", "#93c5fd", "#e0f2fe"])
            .blend(land.visualize(palette=["#1f2937"])))


def coverage_figure():
    """One panel per satellite, so each footprint can be checked on its own."""
    fig, axes = plt.subplots(len(names), 1, figsize=(8, 2.35 * len(names)))
    for ax, n in zip(axes, names):
        url = footprint(GEO[n]).getThumbURL({"region": world, "dimensions": 1100, "format": "png",
                                              "crs": "EPSG:4326"})
        img = Image.open(io.BytesIO(requests.get(url, timeout=600).content))
        ax.imshow(img, extent=[-180, 180, -75, 75])
        ax.axvline(GEO[n], color="#dc2626", lw=0.8, ls="--")
        ax.set_title(f"{n}, above {abs(GEO[n]):g}°{'E' if GEO[n] >= 0 else 'W'}", loc="left", fontsize=9)
        ax.set_xticks(range(-180, 181, 60)); ax.set_yticks([-60, -30, 0, 30, 60])
        ax.set_xticklabels([f"{abs(v)}°{'E' if v > 0 else ('W' if v < 0 else '')}" for v in range(-180, 181, 60)],
                           fontsize=7)
        ax.set_yticklabels([f"{abs(v)}°{'N' if v > 0 else ('S' if v < 0 else '')}" for v in [-60, -30, 0, 30, 60]],
                           fontsize=7)
        ax.grid(color="#64748b", alpha=0.35, lw=0.5)
    fig.suptitle("What each geostationary satellite can see: shaded where its viewing angle is under 70°;\n"
                 "darker blue = more directly below. Of these five, only Himawari-9 and GK-2A see Indonesia",
                 x=0.01, ha="left", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig


# ---------------------------------------------------------------------------
# 2. Where the satellites are, to scale
# ---------------------------------------------------------------------------
def orbits_figure():
    orbits = [("Low Earth orbit: Landsat, Sentinel (about 700 km)", 700, "#64748b"),
              ("GLONASS (19 100 km)", 19100, "#0891b2"),
              ("GPS (20 200 km)", 20200, "#2563eb"),
              ("BeiDou MEO (21 528 km)", 21528, "#7c3aed"),
              ("Galileo (23 222 km)", 23222, "#16a34a"),
              ("Geostationary: Himawari, GK-2A, GOES (35 786 km)", 35786, "#dc2626")]
    fig, ax = plt.subplots(figsize=(8.2, 7.2))
    ax.add_patch(Circle((0, 0), RE, color="#bfdbfe", zorder=3))
    ax.text(0, 0, "Earth", ha="center", va="center", fontsize=9, zorder=4)
    for label, h, col in orbits:
        ax.add_patch(Circle((0, 0), RE + h, fill=False, ec=col, lw=1.6 if h > 1000 else 2.4,
                            label=label))
    ax.legend(frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(0.98, 1.0))
    lim = 44000
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.set_axis_off()
    ax.set_title("Navigation satellites sit between the low imagers and the geostationary ring",
                 loc="left", fontsize=10.5)
    ax.text(-lim, -lim, "To scale. Radius of the Earth 6 378 km.", fontsize=8, color="#64748b")
    return fig


# ---------------------------------------------------------------------------
# 3. Jakarta, New Year's Eve 2019: rain hour by hour from GSMaP
# ---------------------------------------------------------------------------
jabo = ee.Geometry.Rectangle([106.4, -6.9, 107.4, -5.9], None, False)
halim = ee.Geometry.Point([106.89, -6.27])
gsmap = (ee.ImageCollection("JAXA/GPM_L3/GSMaP/v8/operational")
         .filterDate("2019-12-31T00:00:00", "2020-01-01T12:00:00")
         .select("hourlyPrecipRateGC"))
# BMKG's day runs 07:00 to 07:00 local time, which is 00:00 to 00:00 UTC
day_total = gsmap.filterDate("2019-12-31T00:00:00", "2020-01-01T00:00:00").sum().clip(jabo)


def hourly_table():
    def one(img):
        v = img.reduceRegion(ee.Reducer.mean(), halim.buffer(5000), 5000).get("hourlyPrecipRateGC")
        return ee.Feature(None, {"utc": img.date().format("YYYY-MM-dd HH:mm"), "mm_per_h": v})
    fc = ee.FeatureCollection(gsmap.map(one)).getInfo()["features"]
    df = pd.DataFrame([f["properties"] for f in fc]).sort_values("utc")
    df["local_wib"] = pd.to_datetime(df.utc) + pd.Timedelta(hours=7)
    return df


def plot_hourly(df):
    fig, ax = plt.subplots(figsize=(9, 3.8))
    t = pd.to_datetime(df.local_wib)
    ax.bar(t, df.mm_per_h, width=0.035, color="#2563eb")
    day = df[(t >= "2019-12-31 07:00") & (t < "2020-01-01 07:00")].mm_per_h.sum()
    ax.text(0, 1.03, "Rain rate (mm per hour), mean of a 5 km circle around Halim, East Jakarta",
            transform=ax.transAxes, fontsize=8.5, color="#475569")
    ax.set_title(f"GSMaP measured {day:.0f} mm at Halim in the BMKG day; the gauge there "
                 f"recorded 377 mm", loc="left", fontsize=10.5, pad=20)
    ax.set_xlabel("Local time (WIB), 31 December 2019 to 1 January 2020")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.autofmt_xdate()
    return fig


HOURLY = hourly_table()


# ---------------------------------------------------------------------------
# 4. Cyclone Senyar from GK-2A, straight from NOAA's open data on AWS
# ---------------------------------------------------------------------------
GK2A = "https://noaa-gk2a-pds.s3.amazonaws.com/AMI/L1B/FD/{ym}/{d}/{h}/gk2a_ami_le1b_ir105_fd020ge_{stamp}.nc"
CACHE = Path(__file__).resolve().parents[2] / "cache" / "gk2a"
SENYAR = ["202511260010", "202511261210", "202511270010"]      # UTC, every 12 hours
BOX = (92.0, 108.0, -3.0, 11.0)                                # lon0, lon1, lat0, lat1


def gk2a_file(stamp):
    """Download one GK-2A 10.5 µm full disk (about 35 MB) once, and keep it."""
    CACHE.mkdir(parents=True, exist_ok=True)
    f = CACHE / f"ir105_{stamp}.nc"
    if not f.exists() or f.stat().st_size < 1_000_000:
        url = GK2A.format(ym=stamp[:6], d=stamp[6:8], h=stamp[8:10], stamp=stamp)
        f.write_bytes(requests.get(url, timeout=600).content)
    return f


def gk2a_lonlat(attrs, rows, cols):
    """CGMS geostationary projection: image line and column to latitude and longitude."""
    x = np.radians((cols - attrs["coff"]) * 2 ** 16 / attrs["cfac"])
    y = np.radians((rows - attrs["loff"]) * 2 ** 16 / attrs["lfac"])
    cx, cy, sx, sy = np.cos(x), np.cos(y), np.sin(x), np.sin(y)
    k = 1.006739501                                  # (equatorial / polar radius) squared
    a = cy ** 2 + k * sy ** 2
    with np.errstate(invalid="ignore"):
        sd = np.sqrt((42164.0 * cx * cy) ** 2 - a * 1737122264.0)
        sn = (42164.0 * cx * cy - sd) / a
        # GK-2A numbers lines from north to south (LFAC < 0), so y > 0 is north here
        s1, s2, s3 = 42164.0 - sn * cx * cy, sn * sx * cy, sn * sy
        lon = np.degrees(np.arctan(s2 / s1)) + np.degrees(attrs["sub_longitude"])
        lat = np.degrees(np.arctan(k * s3 / np.sqrt(s1 ** 2 + s2 ** 2)))
    return lon, lat


def gk2a_tb(stamp):
    """Brightness temperature (K) of the 10.5 µm band over BOX."""
    with h5py.File(gk2a_file(stamp), "r") as f:
        at = {k: float(np.ravel(f.attrs[k])[0]) for k in
              ("coff", "loff", "cfac", "lfac", "sub_longitude", "DN_to_Radiance_Gain",
               "DN_to_Radiance_Offset", "Teff_to_Tbb_c0", "Teff_to_Tbb_c1", "Teff_to_Tbb_c2")}
        # Find the window of the full disk that covers BOX on a coarse grid first
        rr, cc = np.mgrid[0:5500:20, 0:5500:20]
        lo, la = gk2a_lonlat(at, rr, cc)
        inside = (lo > BOX[0]) & (lo < BOX[1]) & (la > BOX[2]) & (la < BOX[3])
        r0, r1 = rr[inside].min(), rr[inside].max() + 20
        c0, c1 = cc[inside].min(), cc[inside].max() + 20
        dn = f["image_pixel_values"][r0:r1, c0:c1] & 0x1FFF        # 13 valid bits
    rows, cols = np.mgrid[r0:r1, c0:c1]
    lon, lat = gk2a_lonlat(at, rows, cols)
    rad = dn * at["DN_to_Radiance_Gain"] + at["DN_to_Radiance_Offset"]   # mW m-2 sr-1 (cm-1)-1
    nu = 1e4 / 10.5                                                       # wavenumber, cm-1
    c1, c2 = 1.191042e-5, 1.4387752                                       # Planck, these units
    with np.errstate(invalid="ignore", divide="ignore"):
        teff = c2 * nu / np.log(1 + c1 * nu ** 3 / rad)
    tb = at["Teff_to_Tbb_c0"] + at["Teff_to_Tbb_c1"] * teff + at["Teff_to_Tbb_c2"] * teff ** 2
    return lon, lat, tb


class HttpFile(io.RawIOBase):
    """A read-only, seekable file over HTTP range requests, cached in 2 MB blocks.

    GK-2A files are chunked inside, so h5py reads only the chunks a window
    needs: tens of megabytes instead of a 500 MB full disk.
    """
    BLOCK = 2 ** 21

    def __init__(self, url):
        self.url, self.pos, self.cache = url, 0, {}
        self.size = int(requests.head(url, timeout=60).headers["Content-Length"])

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = {0: off, 1: self.pos + off, 2: self.size + off}[whence]
        return self.pos

    def _block(self, i):
        if i not in self.cache:
            a = i * self.BLOCK
            b = min(a + self.BLOCK, self.size) - 1
            self.cache[i] = requests.get(self.url, headers={"Range": f"bytes={a}-{b}"},
                                         timeout=120).content
        return self.cache[i]

    def readinto(self, buf):
        n = min(len(buf), self.size - self.pos)
        out, p = bytearray(), self.pos
        while len(out) < n:
            i, o = divmod(p, self.BLOCK)
            chunk = self._block(i)[o:o + n - len(out)]
            out += chunk
            p += len(chunk)
        buf[:n] = out
        self.pos += n
        return n


VIS = "https://noaa-gk2a-pds.s3.amazonaws.com/AMI/L1B/FD/202610/13/05/gk2a_ami_le1b_{b}_{r}ge_202610130500.nc"
SMOKE_BOX = (109.0, 117.0, -4.5, 1.5)


def gk2a_albedo(band, res, step):
    """Reflectance (albedo) of one visible band over SMOKE_BOX, sampled every `step` pixels."""
    hf = HttpFile(VIS.format(b=band, r=res))
    f = h5py.File(hf, "r")
    at = {k: float(np.ravel(f.attrs[k])[0]) for k in
          ("coff", "loff", "cfac", "lfac", "sub_longitude", "DN_to_Radiance_Gain",
           "DN_to_Radiance_Offset", "Radiance_to_Albedo_c")}
    n = f["image_pixel_values"].shape[0]
    rr, cc = np.mgrid[0:n:n // 275, 0:n:n // 275]
    lo, la = gk2a_lonlat(at, rr, cc)
    inside = ((lo > SMOKE_BOX[0]) & (lo < SMOKE_BOX[1]) & (la > SMOKE_BOX[2]) & (la < SMOKE_BOX[3]))
    r0, r1 = rr[inside].min(), rr[inside].max() + n // 275
    c0, c1 = cc[inside].min(), cc[inside].max() + n // 275
    dn = f["image_pixel_values"][r0:r1, c0:c1] & 0x1FFF
    f.close()
    hf.close()
    if step > 1:                                  # 0.5 km red band to the 1 km grid
        h, w = (dn.shape[0] // step) * step, (dn.shape[1] // step) * step
        dn = dn[:h, :w].reshape(h // step, step, w // step, step).mean(axis=(1, 3))
    alb = (dn * at["DN_to_Radiance_Gain"] + at["DN_to_Radiance_Offset"]) * at["Radiance_to_Albedo_c"]
    rows, cols = np.mgrid[r0:r0 + alb.shape[0] * step:step, c0:c0 + alb.shape[1] * step:step]
    return alb, (at, rows, cols)


def smoke_figure():
    r, (at, rows, cols) = gk2a_albedo("vi006", "fd005", 2)
    g, _ = gk2a_albedo("vi005", "fd010", 1)
    b, _ = gk2a_albedo("vi004", "fd010", 1)
    h, w = min(r.shape[0], g.shape[0], b.shape[0]), min(r.shape[1], g.shape[1], b.shape[1])
    rgb = np.dstack([r[:h, :w], g[:h, :w], b[:h, :w]])
    rgb = np.clip(rgb / 0.55, 0, 1) ** (1 / 2.0)                 # simple stretch and gamma
    lon, lat = gk2a_lonlat(at, rows[:h, :w] + 0.0, cols[:h, :w] + 0.0)
    fires = (ee.Image(ee.ImageCollection("FIRMS").filterDate("2026-10-13", "2026-10-14").max())
             .select("T21").gt(0).selfMask().rename("fire")
             .reduceToVectors(geometry=ee.Geometry.Rectangle([SMOKE_BOX[0], SMOKE_BOX[2], SMOKE_BOX[1],
                                                              SMOKE_BOX[3]], None, False),
                              scale=1000, geometryType="centroid", maxPixels=1e9)
             .getInfo()["features"])
    fx = [ft["geometry"]["coordinates"][0] for ft in fires]
    fy = [ft["geometry"]["coordinates"][1] for ft in fires]
    fig, ax = plt.subplots(figsize=(9, 7.2))
    ax.pcolormesh(lon, lat, rgb[:, :, 0], color=rgb.reshape(-1, 3), shading="auto", rasterized=True)
    ax.scatter(fx, fy, s=3, color="#ff2d2d", lw=0, label=f"FIRMS fire detections that day ({len(fx)})")
    ax.set_xlim(SMOKE_BOX[0], SMOKE_BOX[1]); ax.set_ylim(SMOKE_BOX[2], SMOKE_BOX[3]); ax.set_aspect("equal")
    ax.set_xticks(range(109, 118, 2)); ax.set_xticklabels([f"{v}°E" for v in range(109, 118, 2)], fontsize=8)
    ax.set_yticks(range(-4, 2, 1))
    ax.set_yticklabels([f"{abs(v)}°{'N' if v > 0 else ('S' if v < 0 else '')}" for v in range(-4, 2)], fontsize=8)
    ax.grid(color="white", alpha=0.35, lw=0.5)
    ax.legend(loc="lower right", fontsize=8, framealpha=0.85, markerscale=3)
    ax.set_title("Smoke from Kalimantan's fires, 13 October 2026, 12:00 WIB: GK-2A true colour",
                 loc="left", fontsize=10.5)
    return fig


def senyar_figure():
    lsib = ee.FeatureCollection("USDOS/LSIB_SIMPLE/2017").filterBounds(
        ee.Geometry.Rectangle([BOX[0], BOX[2], BOX[1], BOX[3]], None, False))
    feats = lsib.map(lambda f: f.simplify(2000)).getInfo()["features"]
    rings = []
    for ft in feats:
        g = ft["geometry"]
        parts = g.get("geometries", [g])
        for part in parts:
            if part["type"] == "Polygon":
                rings.append(np.array(part["coordinates"][0]))
            elif part["type"] == "MultiPolygon":
                rings += [np.array(pp[0]) for pp in part["coordinates"]]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.6), sharey=True)
    for ax, stamp in zip(axes, SENYAR):
        lon, lat, tb = gk2a_tb(stamp)
        m = ax.pcolormesh(lon, lat, tb - 273.15, cmap="turbo_r", vmin=-85, vmax=30, shading="auto",
                          rasterized=True)
        for ring in rings:
            ax.plot(ring[:, 0], ring[:, 1], color="white", lw=0.5)
        ax.set_xlim(BOX[0], BOX[1]); ax.set_ylim(BOX[2], BOX[3]); ax.set_aspect("equal")
        t = pd.Timestamp(stamp, tz="UTC").tz_convert("Asia/Jakarta")
        ax.set_title(t.strftime("%d %b %Y, %H:%M WIB"), loc="left", fontsize=9.5)
        ax.grid(color="white", alpha=0.3, lw=0.5)
        ax.set_xticks(range(92, 109, 4)); ax.set_xticklabels([f"{v}°E" for v in range(92, 109, 4)],
                                                             fontsize=7.5)
    axes[0].set_yticks(range(-2, 11, 4)); axes[0].set_yticklabels(
        [f"{abs(v)}°{'N' if v >= 0 else 'S'}" for v in range(-2, 11, 4)], fontsize=7.5)
    cb = fig.colorbar(m, ax=axes, orientation="horizontal", fraction=0.05, pad=0.1, aspect=50)
    cb.set_label("Cloud-top (brightness) temperature at 10.5 µm (°C); coldest = tallest storm cloud",
                 fontsize=8)
    fig.suptitle("Cyclone Senyar crosses northern Sumatra: GK-2A saw it every 10 minutes",
                 x=0.01, ha="left", fontsize=11)
    return fig


def products():
    return [
        {"kind": "figure", "name": "p7-geo-coverage", "figure": coverage_figure,
         "caption": "Each geostationary satellite sees the same disk, a little less than half the "
                    "Earth, always from the same place above the equator (red dashed line). Beyond "
                    "about 70° from vertical the view is too slanted to use, and the poles are seen "
                    "by none. Of these five, only Himawari-9 and GK-2A look into Indonesia usefully."},
        {"kind": "table", "name": "p7-pixel-stretch", "data": cities_table,
         "caption": "A 2 km infrared pixel under Himawari-9 (140.7° E) grows with distance and "
                    "with the slant of the view. Western Indonesia is seen at about 40°, so its "
                    "pixels are around 3 km across and longer towards the edge of the disk."},
        {"kind": "figure", "name": "p7-orbits", "figure": orbits_figure,
         "caption": "Orbits to scale. Imaging satellites fly a few hundred kilometres up; "
                    "navigation constellations about 19 000 to 23 000 km; weather satellites "
                    "that never move across the sky, 35 786 km above the equator."},
        {"kind": "figure", "name": "p7-senyar", "figure": senyar_figure,
         "caption": "GK-2A's 10.5 µm band, three of the 144 full disks it took each day, read "
                    "straight from the open archive on AWS (noaa-gk2a-pds). The storm's coldest "
                    "cloud tops, below -70 °C, sit over the Strait of Malacca and Aceh as Senyar "
                    "makes landfall around midnight on 26 November 2025."},
        {"kind": "figure", "name": "p7-smoke", "figure": smoke_figure,
         "caption": "GK-2A's blue, green and red bands (1 km, the red averaged from 0.5 km) at "
                    "noon on 13 October 2026, read from the open archive on AWS, with that day's "
                    "MODIS fire detections from FIRMS. Smoke is the grey-brown haze, thicker than "
                    "cloud is white; the same view comes every 10 minutes."},
        {"kind": "map", "name": "p7-gsmap-day", "image": day_total, "region": jabo,
         "vis": {"min": 0, "max": 250, "palette": ["#f8fafc", "#93c5fd", "#2563eb", "#7c3aed", "#db2777"]},
         "legend": "Rain, 31 Dec 2019 07:00 to 1 Jan 2020 07:00 WIB (mm)",
         "title": "Jakarta's New Year rain, GSMaP", "source": "JAXA GSMaP v8, gauge calibrated. GEE.",
         "caption": "One day of rain as GSMaP estimated it, from microwave radiometers on low "
                    "satellites moved along by cloud motion seen from geostationary orbit, then "
                    "adjusted with rain gauges."},
        {"kind": "chart", "name": "p7-gsmap-hourly", "data": HOURLY, "plot": plot_hourly,
         "caption": "The same storm hour by hour at Halim. Satellite rain is an average over "
                    "about 10 km, so it smooths the peak a single gauge catches."},
    ]


if __name__ == "__main__":
    ee.Initialize()
    print(cities_table())
    print(HOURLY.mm_per_h.sum())
