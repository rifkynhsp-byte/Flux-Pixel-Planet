#| title: Oil slicks and ships from Sentinel-1 radar (R)
#| description: Finds the 2018 Balikpapan oil slick as dark spots on a Sentinel-1 image, tells it from look-alikes by shape and contrast, detects ships with a CFAR test, and turns twelve-day snapshots of the Malacca Strait into ships per day.

# Fullstack Remote Sensing | Code by Rifky Nauval Hendrawan, Remote Sensing Specialist
# MIT licence: free to use and adapt; please keep this credit line.

# CHAPTER 77 | Oil and ships under radar  (R)
# Data in the book repository:
#   data/ch77_balikpapan_vv_20180402.tif   Sentinel-1 VV (dB x 100) and a sea mask, 25 m, 2 April 2018 05:50 local
#   data/ch77_malacca_vh_chip.tif          Sentinel-1 VH (dB x 100), 10 m, one 2025 pass over the shipping lanes
#   data/ch77_malacca_ships_2025.csv       ships detected by Earth Engine on every 2025 pass (JavaScript/Python tabs)

library(terra)
library(dplyr)
library(ggplot2)

# PART 1. The slick: dark spots on the sea -----------------------------------------------------
r <- rast("data/ch77_balikpapan_vv_20180402.tif")
vv <- r[[1]] / 100                 # band 1: VV in dB x 100
sea <- r[[2]] == 1                 # band 2: 1 = sea
vv <- mask(vv, sea, maskvalues = FALSE)

# Speckle: radar images are grainy; a 5 x 5 mean (125 m) smooths it before thresholding.
vv_f <- focal(vv, w = 5, fun = "mean", na.rm = TRUE)

# Oil damps the small waves that scatter radar back, so slicks are darker than the sea AROUND them.
# The background must reach clean sea beyond the slick: a big slick is 10-20 km wide, and a small window
# would sit inside it and call it normal. Median within ~10 km, computed at 250 m and brought back to 25 m.
bg <- disagg(focal(aggregate(vv_f, 10, fun = "median", na.rm = TRUE), w = 41, fun = "median", na.rm = TRUE), 10)
bg <- resample(bg, vv_f)
dark <- (vv_f < bg - 3) & (vv_f < -24)          # 3 dB darker than the local sea AND dark in absolute terms
dark <- mask(dark, sea, maskvalues = FALSE)

# Patches: connected dark areas; drop specks under 0.5 km2 (800 pixels of 25 m).
p <- patches(classify(dark, cbind(0, NA)), directions = 8)
px_km2 <- prod(res(p)) / 1e6
st <- as.data.frame(freq(p)) |> transmute(id = value, area_km2 = count * px_km2) |> filter(area_km2 >= 0.5)

# Features that separate oil from look-alikes (Brekke & Solberg 2005): size, contrast with the sea, shape.
contrast <- zonal(bg - vv_f, p, fun = "mean", na.rm = TRUE); names(contrast) <- c("id", "contrast_db")
pol <- as.polygons(p) |> (\(x) x[x$patches %in% st$id])()
st <- st |> left_join(contrast, by = "id") |>
  left_join(data.frame(id = pol$patches, perim_km = perim(pol) / 1000), by = "id") |>
  mutate(compactness = 4 * pi * area_km2 / perim_km^2,              # 1 = circle, near 0 = thin or ragged
         class = case_when(area_km2 >= 5 & contrast_db >= 5 ~ "likely oil",
                           contrast_db < 4 ~ "look-alike: low contrast (calm water)",
                           TRUE ~ "check: small or weak"))
print(st |> arrange(desc(area_km2)) |> head(10))
cat(sprintf("Dark area classed 'likely oil': %.1f km2 in %d patches\n",
            sum(st$area_km2[st$class == "likely oil"]), sum(st$class == "likely oil")))

# Map in longitude/latitude: detected oil in orange, look-alikes and small patches in blue.
cls <- classify(p, rbind(cbind(st$id[st$class == "likely oil"], 2), cbind(st$id[st$class != "likely oil"], 1)), others = NA)
vv_ll <- project(aggregate(vv_f, 4, fun = "mean", na.rm = TRUE), "EPSG:4326")
cl_ll <- project(aggregate(cls, 4, fun = "modal", na.rm = TRUE), vv_ll, method = "near")
df <- as.data.frame(c(vv_ll, cl_ll), xy = TRUE); names(df)[3:4] <- c("vv", "cls")
df <- df[!is.na(df$vv), ]
ggplot(df, aes(x, y)) + geom_raster(aes(fill = vv)) +
  scale_fill_gradient(low = "black", high = "white", limits = c(-30, -15), oob = scales::squish, name = "VV (dB)") +
  geom_tile(data = subset(df, cls == 2), fill = "#e8710a", alpha = 0.55) +
  geom_tile(data = subset(df, cls == 1), fill = "#56b4e9", alpha = 0.8) +
  coord_quickmap(expand = FALSE) +
  labs(title = sprintf("Balikpapan Bay, 2 April 2018: %.0f km² of likely oil (orange)", sum(st$area_km2[st$class == "likely oil"])),
       x = "Longitude (°E)", y = "Latitude (°)", caption = "Sentinel-1 VV, 25 m, 05:50 local. Blue: dark patches too small or too weak to call. White: land.") +
  theme_minimal() + theme(panel.grid = element_line(colour = "grey70", linewidth = 0.2))

# PART 2. Ships: a CFAR test on every pass ------------------------------------------------------
# A ship is a hard, metal target: in VH it is 10-20 dB brighter than the sea around it. CFAR (constant false
# alarm rate) compares each pixel with the LOCAL sea, so the test adapts to wind: a pixel is a ship when it
# is more than 10 dB above the median sea within ~1.5 km.
detect_ships <- function(f, thr_db = 10, abs_db = -18, min_px = 3) {
  r <- rast(f); vh <- r[[1]] / 100; sea <- r[[2]] == 1
  # keep sea at least ~1 km from the coast: shrink the mask at 200 m, then bring it back to 20 m
  sea_core <- resample(focal(aggregate(sea, 10, fun = "min"), w = 11, fun = "min", na.rm = TRUE) == 1, vh, method = "near")
  lin <- 10^(vh / 10)
  bg <- resample(focal(aggregate(lin, 5, fun = "median", na.rm = TRUE), w = 15, fun = "median", na.rm = TRUE), lin)
  # Two tests: bright RELATIVE to the local sea, and bright in ABSOLUTE terms. Calm sea in VH sits near the
  # sensor's noise floor, where noise between sub-swaths can pass a relative test alone; a steel hull is
  # brighter than about -18 dB whatever the sea does.
  hit <- (lin / bg > 10^(thr_db / 10)) & (vh > abs_db) & sea_core
  p <- patches(classify(hit, cbind(0, NA)), directions = 8)
  if (all(is.na(values(p)))) return(NULL)
  n <- as.data.frame(freq(p)) |> transmute(id = value, npx = count) |> filter(npx >= min_px)
  xy <- as.data.frame(p, xy = TRUE) |> rename(id = 3) |> filter(id %in% n$id) |>
    group_by(id) |> summarise(x = mean(x), y = mean(y), .groups = "drop") |> left_join(n, by = "id")
  ll <- project(vect(xy, geom = c("x", "y"), crs = crs(p)), "EPSG:4326")
  pk <- zonal(vh, p, fun = "max", na.rm = TRUE); names(pk) <- c("id", "peak_db")
  xy <- left_join(xy, pk, by = "id")
  data.frame(file = basename(f), lon = geom(ll)[, "x"], lat = geom(ll)[, "y"], npx = xy$npx, peak_db = round(xy$peak_db, 1))
}

# The 29 passes of 2025 are ~16 MB each, so they are not in the book repository. Earth Engine exports
# them (JavaScript/Python tabs); here we read the detections it made, or rebuild them if you have the scenes.
scene_dir <- Sys.getenv("CH77_SCENES", "")
if (nzchar(scene_dir) && dir.exists(scene_dir)) {
  files <- list.files(scene_dir, pattern = "^malacca_.*\\.tif$", full.names = TRUE)
  ships <- do.call(rbind, lapply(files, detect_ships))
  ships$date <- as.Date(substr(ships$file, 9, 16), "%Y%m%d")
  write.csv(ships[, c("date", "lon", "lat", "npx", "peak_db")], "data/ch77_malacca_ships_2025.csv", row.names = FALSE)
}
ships <- read.csv("data/ch77_malacca_ships_2025.csv") |> mutate(date = as.Date(date))

# PART 3. From snapshots to ships per day --------------------------------------------------------
# Fixed bright targets (platforms, buoys, lights, ships at long anchor) appear pass after pass in the same
# place. Count passes per 500 m cell; a cell lit in more than a quarter of all passes is not traffic.
n_pass <- n_distinct(ships$date)
ships <- ships |> mutate(cx = round(lon / 0.0045), cy = round(lat / 0.0045))
fixed <- ships |> distinct(date, cx, cy) |> count(cx, cy, name = "passes") |> filter(passes > n_pass / 4)
moving <- ships |> anti_join(fixed, by = c("cx", "cy"))
per_pass <- moving |> count(date, name = "ships") |> right_join(data.frame(date = unique(ships$date)), by = "date") |>
  mutate(ships = coalesce(ships, 0L)) |> arrange(date)
cat(sprintf("%d passes; fixed targets removed: %d cells; ships per pass: median %.0f (range %d-%d)\n",
            n_pass, nrow(fixed), median(per_pass$ships), min(per_pass$ships), max(per_pass$ships)))

# Traffic flow = density x speed. Ships in the gate / length of their path through it = ships per km;
# times speed (km/h) and 24 h = ships crossing per day. The gate spans 2.75-3.05 deg N (33.2 km north-south);
# traffic runs NW-SE at about 135 deg, so each ship's path inside it is 33.2 / cos(45 deg) = 46.9 km.
L_km <- 33.2 / cos(45 * pi / 180)
set.seed(77)
sim <- replicate(5000, {
  n <- sample(per_pass$ships, replace = TRUE)               # resample the passes
  v <- runif(1, 10, 16) * 1.852                             # a typical transit speed, 10-16 knots, in km/h
  mean(n) / L_km * v * 24
})
q <- quantile(sim, c(0.05, 0.5, 0.95))
cat(sprintf("Ships per day through the gate: %.0f (90%% interval %.0f-%.0f). STRAITREP 2025: 281 per day (>300 GT).\n", q[2], q[1], q[3]))

# What counts as a ship? Radar sees every hull, from fishing boats to tankers; STRAITREP counts only vessels over
# 300 GT that report. A larger minimum target is a rough stand-in for a larger ship (radar spreads a hull's return,
# so bright pixels are not length), and the answer moves with it:
sens <- lapply(c(3, 5, 10, 20, 40), function(k) {
  s <- moving |> filter(npx >= k) |> count(date, name = "n") |> right_join(data.frame(date = unique(ships$date)), by = "date") |>
    mutate(n = coalesce(n, 0L))
  data.frame(min_pixels = k, bright_area_m2 = k * 400, ships_per_pass = median(s$n), ships_per_day_13kn = round(mean(s$n) / L_km * 13 * 1.852 * 24))
})
print(do.call(rbind, sens))

ggplot(per_pass, aes(date, ships)) + geom_col(fill = "#1f5fd6", width = 6) +
  geom_hline(yintercept = mean(per_pass$ships), linetype = "dashed") +
  labs(title = "Ships in the gate at each Sentinel-1 pass, 2025 (19:34 local time, every 12 days)",
       x = NULL, y = "ships detected", caption = "CFAR on VH; fixed targets removed. Dashed: mean.") + theme_minimal()

ggplot(moving, aes(lon, lat)) + geom_bin2d(binwidth = c(0.01, 0.01)) +
  scale_fill_viridis_c(option = "magma", trans = "log10", name = "detections\n(2025)") +
  coord_quickmap(expand = FALSE) +
  labs(title = "Where the ships were: the shipping lanes of the Malacca Strait appear from detections alone",
       x = "Longitude (°E)", y = "Latitude (°N)") + theme_minimal()
