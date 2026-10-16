/**
 * CHAPTER 77 | Oil and ships under radar
 * ---------------------------------------------------------------------------
 * Fullstack Remote Sensing | Code by Rifky Nauval Hendrawan, Remote Sensing Specialist
 * MIT licence: free to use and adapt; please keep this credit line.
 *
 * Part 1. The Balikpapan Bay oil spill, 31 March 2018: Sentinel-1 VV on 2 April 05:50 local,
 *         dark spots against the surrounding sea, and the slick area.
 * Part 2. Ships in the Malacca Strait: a CFAR test on one 2025 pass over a gate across the strait.
 *         Counting all 29 passes of 2025 is too heavy for an interactive request; the R and Python
 *         tabs do it from downloaded scenes and turn the counts into ships per day.
 */

// ---------- PART 1. The slick ---------------------------------------------------------------
var bay = ee.Geometry.Rectangle([116.75, -1.42, 117.15, -1.12], null, false);
var sea = ee.ImageCollection('MODIS/006/MOD44W').sort('system:time_start', false).first()
  .select('water_mask').eq(1);

var vv = ee.ImageCollection('COPERNICUS/S1_GRD')
  .filterBounds(bay).filterDate('2018-04-01T21:00', '2018-04-01T23:00')
  .filter(ee.Filter.eq('instrumentMode', 'IW'))
  .select('VV').mosaic().updateMask(sea);

// Speckle: a 125 m mean. Background: the sea within ~10 km (computed at 250 m, ignoring land).
// The window must reach clean sea beyond the slick; a 2 km window sits inside a 15 km slick.
var vvSmooth = vv.focalMean(62, 'circle', 'meters');
var P250 = ee.Projection('EPSG:32750').atScale(250);
var background = vvSmooth.reduceResolution(ee.Reducer.median(), false, 256).reproject(P250)
  .focalMedian(5000, 'circle', 'meters');
var dark = vvSmooth.lt(background.subtract(3)).and(vvSmooth.lt(-24)).selfMask();

// Clean up: keep connected dark areas of at least 0.5 km2 (800 pixels of 25 m).
var big = dark.updateMask(dark.connectedPixelCount(1024, true).gte(800));
var areaKm2 = big.multiply(ee.Image.pixelArea()).divide(1e6)
  .reduceRegion({reducer: ee.Reducer.sum(), geometry: bay, scale: 25, maxPixels: 1e9, tileScale: 4});
print('Dark area >= 0.5 km2 (km2):', areaKm2);   // about 54-56 km2; the R tab also tests contrast and shape

Map.centerObject(bay, 11);
Map.addLayer(vvSmooth, {min: -28, max: -10}, 'Sentinel-1 VV, 2 Apr 2018 (dB)');
Map.addLayer(big, {palette: ['ff7f00']}, 'Dark spots >= 0.5 km2 (candidate oil)');

// ---------- PART 2. Ships: CFAR on one pass ---------------------------------------------------
var gate = ee.Geometry.Rectangle([100.2, 2.75, 101.15, 3.05], null, false);
var vh = ee.ImageCollection('COPERNICUS/S1_GRD')
  .filterBounds(gate).filterDate('2025-01-09', '2025-01-10')
  .filter(ee.Filter.eq('instrumentMode', 'IW'))
  .filter(ee.Filter.eq('orbitProperties_pass', 'ASCENDING'))
  .select('VH').mosaic();
var seaCore = sea.reproject(ee.Projection('EPSG:32647').atScale(250)).focalMin(4);   // 1 km off the coast

// CFAR: brighter than the local sea by 10 dB (relative) AND brighter than -18 dB (absolute).
var lin = ee.Image(10).pow(vh.divide(10));
var bgSea = lin.reduceResolution(ee.Reducer.median(), false, 1024)
  .reproject(ee.Projection('EPSG:32647').atScale(100)).focalMedian(1500, 'circle', 'meters');
var ship = lin.divide(bgSea).gt(10).and(vh.gt(-18)).and(seaCore).selfMask();
var shipObj = ship.updateMask(ship.connectedPixelCount(200, true).gte(3));

Map.addLayer(vh, {min: -28, max: -8}, 'Sentinel-1 VH, 9 Jan 2025 (dB)', false);
Map.addLayer(shipObj, {palette: ['00ffff']}, 'Ship detections (CFAR)', false);

// Count the ships of this pass in a 20 km wide slice of the lanes (small enough to answer quickly).
var slice = ee.Geometry.Rectangle([100.75, 2.75, 100.95, 3.05], null, false);
var n = shipObj.reduceToVectors({geometry: slice, scale: 20, geometryType: 'centroid',
                                 eightConnected: true, maxPixels: 1e9, tileScale: 4}).size();
print('Ships in the 100.75-100.95 E slice on 9 Jan 2025:', n);
