//| title: Geostationary and navigation satellites
//| description: How directly Himawari-9 sees each place, and Jakarta's New Year 2020 rain from GSMaP, hour by hour.

// Fullstack Remote Sensing | Code by Rifky Nauval Hendrawan, Remote Sensing Specialist
// MIT licence: free to use and adapt; please keep this credit line.

/**
 * PRINCIPLES P7 | Satellites that stay put
 * ---------------------------------------------------------------------------
 * 1. The viewing (zenith) angle from Himawari-9, 35 786 km above 140.7° E,
 *    computed for every pixel from the orbit geometry.
 * 2. GSMaP: rain from microwave radiometers on low satellites, moved between
 *    their passes with cloud motion seen from geostationary orbit.
 * The GK-2A images of Cyclone Senyar are read from AWS in the Python tab.
 */

var RE = 6378.137, RS = RE + 35786.0, SUBLON = 140.7;   // Himawari-9

// 1. Viewing angle from Himawari-9.
var ll = ee.Image.pixelLonLat();
var lat = ll.select('latitude').multiply(Math.PI / 180);
var dlon = ll.select('longitude').subtract(SUBLON).multiply(Math.PI / 180);
var cosg = lat.cos().multiply(dlon.cos());
var d = ee.Image(RE * RE + RS * RS).subtract(cosg.multiply(2 * RE * RS)).sqrt();
var vza = cosg.acos().sin().multiply(RS).divide(d).asin().multiply(180 / Math.PI)
  .updateMask(cosg.gt(RE / RS));                       // beyond the limb: not seen at all
Map.setCenter(118, -2, 4);
Map.addLayer(vza.updateMask(vza.lt(70)), {min: 0, max: 70, palette: ['1d4ed8', '93c5fd', 'e0f2fe']},
  'Viewing angle from Himawari-9 (degrees)');

// 2. Jakarta, 31 December 2019 to 1 January 2020: one BMKG day of rain.
var jabo = ee.Geometry.Rectangle([106.4, -6.9, 107.4, -5.9], null, false);
var gsmap = ee.ImageCollection('JAXA/GPM_L3/GSMaP/v8/operational')
  .filterDate('2019-12-31T00:00:00', '2020-01-01T12:00:00').select('hourlyPrecipRateGC');
var day = gsmap.filterDate('2019-12-31T00:00:00', '2020-01-01T00:00:00').sum().clip(jabo);
Map.addLayer(day, {min: 0, max: 250, palette: ['f8fafc', '93c5fd', '2563eb', '7c3aed', 'db2777']},
  'GSMaP rain, 24 h to 07:00 WIB 1 Jan 2020 (mm)');

var halim = ee.Geometry.Point([106.89, -6.27]).buffer(5000);
print(ui.Chart.image.series(gsmap, halim, ee.Reducer.mean(), 5000)
  .setOptions({title: 'GSMaP rain rate at Halim (mm per hour, UTC)', legend: 'none'}));
print('GSMaP total at Halim for the BMKG day (mm):',
  day.reduceRegion(ee.Reducer.mean(), halim, 5000).get('hourlyPrecipRateGC'));
print('BMKG gauge at Halim recorded 377 mm.');
