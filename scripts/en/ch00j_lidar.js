//| title: LiDAR, and passive against active
//| description: GEDI canopy height around Palangka Raya and the edge of Sebangau, good-quality footprints only, and its distribution by Dynamic World class.

// Fullstack Remote Sensing | Code by Rifky Nauval Hendrawan, Remote Sensing Specialist
// MIT licence: free to use and adapt; please keep this credit line.

/**
 * PRINCIPLES P8 | LiDAR from orbit
 * ---------------------------------------------------------------------------
 * GEDI fires a 1064 nm laser from the International Space Station and records
 * the full returned waveform of each 25 m footprint. RH98 is the height below
 * which 98 % of the energy came back: the usual measure of canopy height.
 * The waveform figure is computed in the Python tab.
 */

var kalteng = ee.Geometry.Rectangle([113.70, -2.45, 114.40, -1.95], null, false);

function good(img) {
  return img.updateMask(img.select('quality_flag').eq(1))
            .updateMask(img.select('degrade_flag').eq(0));
}

var gedi = ee.ImageCollection('LARSE/GEDI/GEDI02_A_002_MONTHLY').filterBounds(kalteng)
  .filterDate('2019-04-01', '2024-01-01').map(good).select('rh98').max().clip(kalteng);

Map.centerObject(kalteng, 10);
Map.addLayer(gedi, {min: 0, max: 40, palette: ['fef9c3', 'a3e635', '16a34a', '14532d']},
  'GEDI RH98, tallest of 2019-2023 (m)');

// Height by land cover: forest against cleared and burned peat.
var dw = ee.ImageCollection('GOOGLE/DYNAMICWORLD/V1').filterBounds(kalteng)
  .filterDate('2022-01-01', '2024-01-01').select('label').mode();
var samples = gedi.rename('rh98').addBands(dw.rename('cls'))
  .stratifiedSample({numPoints: 400, classBand: 'cls', region: kalteng, scale: 25, seed: 1});
print(ui.Chart.feature.histogram(samples.filter(ee.Filter.eq('cls', 1)), 'rh98', 40)
  .setOptions({title: 'GEDI RH98, Dynamic World trees'}));
print(ui.Chart.feature.histogram(samples.filter(ee.Filter.eq('cls', 5)), 'rh98', 40)
  .setOptions({title: 'GEDI RH98, Dynamic World shrub and scrub'}));
