# eclipse-heatmap

A global heat map of **days until the next visible solar eclipse**, computed
from the JPL DE440 ephemeris via Skyfield.
![example output](output/days_until_next_eclipse.png)

## What it does

For every point on a lat/lon grid, we find the first future solar eclipse
(partial, annular, or total) with any visibility at all, and record how
many days from today that is (`days_until_next_eclipse.csv`/`.tif`/`.npy`).

The PNG map shows more than just that first eclipse: every eclipse's
footprint is painted onto every point it touches, with opacity set by
magnitude (grazing partial = nearly transparent, total = fully opaque)
and color set by chronological order. Where two eclipses' footprints
overlap the same point, their colors blend (Porter-Duff "over"
compositing) instead of one simply replacing the other.

The search keeps finding eclipses further into the future, rendering the
map after every single one, until either every grid point is covered or
you stop it.

## Install

```bash
python3.11 -m venv .venv        # 3.12 also fine
source .venv/bin/activate
pip install -e .
```

## Run

```bash
python main.py                                            # full open-ended run, all CPU cores
python main.py --resolution 1.0 --end-date 2030-12-31     # bounded, faster
python main.py --resolution 2.0 --lat-bounds 60 68 --lon-bounds -25 -13 \
    --start-date 2026-08-01 --end-date 2026-08-20          # quick regional test
```

### CLI options

| Flag | Default | Meaning |
|---|---|---|
| `--resolution` | `0.25` | Grid spacing, degrees |
| `--start-date` | today | Search start date, `YYYY-MM-DD` |
| `--end-date` | none | Search end date; omit to run until full coverage or Ctrl+C |
| `--time-step-seconds` | `60` | Time resolution per eclipse window |
| `--workers` | all cores | Worker processes; `1` disables multiprocessing |
| `--data-dir` | `data/` | Ephemeris cache directory |
| `--ephemeris-filename` | `de440.bsp` | Kernel file; `de440s.bsp` is a smaller (~32MB) alternative |
| `--output-dir` | `output/` | Where results are written |
| `--max-eclipses` | none | Stop after N events (debugging) |
| `--lat-bounds MIN MAX` | `-90 90` | Restrict grid latitude |
| `--lon-bounds MIN MAX` | `-180 180` | Restrict grid longitude |
| `--log-level` | `INFO` | Logging verbosity |

## Outputs

Written to `--output-dir`:

| File | Contents |
|---|---|
| `days_until_next_eclipse.npy` | 2-D array, north-up, NaN = not covered yet |
| `days_until_next_eclipse.tif` | Same data as GeoTIFF, EPSG:4326 |
| `days_until_next_eclipse.csv` | One row per grid point: lat, lon, days, date, type, magnitude, eclipse rank |
| `days_until_next_eclipse.png` | Robinson-projection heat map; color = time, opacity = magnitude, overlapping eclipses blend |

## Layout

```text
main.py                              thin shim -> eclipse_heatmap.cli.main
pyproject.toml                       package metadata, dependencies, console script
src/eclipse_heatmap/
  cli.py                             argument parsing
  pipeline.py                        workflow: ephemeris -> grid -> sweep -> save
  models/                            data structures: GridSpec, SolarEclipseEvent, EclipseType
  science/                           astronomy: geometry, eclipse search, visibility sweep
  rendering/                         output writers: raster, table, heatmap
  utils/                             generic helpers: logging, geo math
```

## Accuracy

- Ephemeris: JPL DE440 (Park et al. 2021) via Skyfield/jplephem.
- Real topocentric geometry per grid point: light-time, aberration, Earth
  orientation, WGS84 observer position.
- No small-angle approximations in the eclipse geometry.

## Limitations

- Atmospheric refraction near the horizon is not modeled.
- "Days until" uses one UTC date per eclipse, not each observer's local date.
- A coarse `--time-step-seconds` can miss brief totality at the edge of a path.
- Earth orientation (ΔT/UT1) for far-future dates is extrapolated, as in
  all long-range eclipse predictions.

## References

- NASA GSFC, *Eclipse Predictions* — https://eclipse.gsfc.nasa.gov/
- Park, R. S. et al. (2021), "The JPL Planetary and Lunar Ephemerides
  DE440 and DE441", *The Astronomical Journal*, 161, 105.
- Meeus, J., *Astronomical Algorithms*, 2nd ed., Willmann-Bell, 1998.
- *Explanatory Supplement to the Astronomical Almanac*, 3rd ed. (2013).
- Rhodes, B., *Skyfield* documentation — https://rhodesmill.org/skyfield/
- Nuñez, J., Anderton, C., Renslow, R. (2018), "Optimizing colormaps with
  consideration for color vision deficiency," *PLOS ONE* 13(7) — cividis colormap.
