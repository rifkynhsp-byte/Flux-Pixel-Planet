//| title: Hyperspectral against multispectral
//| description: One EMIT scene over Saguling and Bandung in true colour, its 285-band spectrum at any clicked point, and the same point as Sentinel-2 sees it.

// Fullstack Remote Sensing | Code by Rifky Nauval Hendrawan, Remote Sensing Specialist
// MIT licence: free to use and adapt; please keep this credit line.

/**
 * PRINCIPLES P6 | Hyperspectral against multispectral
 * ---------------------------------------------------------------------------
 * EMIT (on the International Space Station) measures 285 narrow bands from
 * 381 to 2493 nm at 60 m. Sentinel-2 measures 13 wide bands. Click the map to
 * compare the two at the same place.
 */

var aoi = ee.Geometry.Rectangle([107.30, -7.05, 107.75, -6.75], null, false);

// 1. The clearest dry-season EMIT pass over the area, cloud masked.
var emit = ee.Image(ee.ImageCollection('NASA/EMIT/L2A/RFL')
  .filter(ee.Filter.eq('system:index', 'EMIT_L2A_RFL_001_20250826T041241_2523802_036')).first());
var bad = emit.select('cloud_flag').or(emit.select('cirrus_flag')).or(emit.select('dilated_cloud_flag'));
var bands = ee.List.sequence(0, 284).map(function (i) {
  return ee.String('reflectance_').cat(ee.Number(i).int().format());
});
var rfl = emit.select(bands).updateMask(bad.not()).clip(aoi);
var wavelengths = ee.List(emit.get('reflectance_wavelengths'));

Map.centerObject(aoi, 11);
Map.addLayer(rfl, {bands: ['reflectance_35', 'reflectance_23', 'reflectance_11'],
  min: 0, max: 0.15, gamma: 1.3}, 'EMIT true colour, 26 Aug 2025');

// 2. Sentinel-2 surface reflectance for the same month, for comparison.
var s2 = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED').filterBounds(aoi)
  .filterDate('2025-08-01', '2025-09-15')
  .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30))
  .median().divide(10000).clip(aoi);
var s2Bands = ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B8A', 'B9', 'B11', 'B12'];
var s2Centres = [443, 490, 560, 665, 705, 740, 783, 842, 865, 945, 1610, 2190];

// 3. Click anywhere: two spectra of the same 60 m spot.
var panel = ui.Panel({style: {width: '420px'}});
panel.add(ui.Label('Click the map: EMIT (285 bands) against Sentinel-2 (12 bands)'));
ui.root.add(panel);

Map.onClick(function (coords) {
  var pt = ee.Geometry.Point([coords.lon, coords.lat]).buffer(30);
  panel.widgets().reset([ui.Label('Lon ' + coords.lon.toFixed(4) + ', lat ' + coords.lat.toFixed(4))]);
  panel.add(ui.Chart.image.regions(rfl, pt, ee.Reducer.mean(), 60, null, wavelengths)
    .setChartType('LineChart')
    .setOptions({title: 'EMIT spectrum', hAxis: {title: 'Wavelength (nm)'},
                 vAxis: {title: 'Reflectance'}, lineWidth: 1, pointSize: 0, legend: 'none'}));
  panel.add(ui.Chart.image.regions(s2.select(s2Bands), pt, ee.Reducer.mean(), 10, null, s2Centres)
    .setChartType('LineChart')
    .setOptions({title: 'Sentinel-2, same place', hAxis: {title: 'Wavelength (nm)'},
                 vAxis: {title: 'Reflectance'}, pointSize: 5, legend: 'none'}));
});
