# Measured nine-zone residential data

`house_data.csv` is redistributed unchanged from Charalampos Vallianos,
*Automated Multi-Zone Model Generation*, v1.0 (2023),
https://doi.org/10.5281/zenodo.10156745 .
Archive source: https://github.com/HarryVallianos/Automated-MultiZone-Model-Generation/tree/v1.0
Archive commit prefix: `132db80`.

Related data/method article: C. Vallianos, A. Athienitis, B. Delcroix,
*Automatic generation of multi-zone RC models using smart thermostat data from homes*,
Energy and Buildings 277 (2022), 112571. https://doi.org/10.1016/j.enbuild.2022.112571

SHA-256 of the unchanged CSV:
`d7da67d1be8e57be1419379f61ddc82fab737fc888141ac48754a898549f7bc8`
The Zenodo ZIP's recorded MD5 was checked before extraction.

## Schema and timing

385 hourly timestamps, 2019-03-30 00:00 through 2019-04-15 00:00,
with no missing observations; no timezone is specified by the source.
`T01_TEMP` ... `T09_TEMP`: measured zone temperature (degrees Celsius).
`T01_Wh` ... `T09_Wh`: heating energy over an hourly interval (Wh).
`Text`: outside temperature (degrees Celsius). `GHI`: global horizontal irradiance (W/m2).
The benchmark uses the heating/weather values on row t to predict the
zone temperatures on row t+1, an explicit hourly alignment convention.
It does not assert sub-hourly timing or real-time availability of these inputs.

## Attribution and licenses

The archived source includes the GNU Affero General Public License v3;
that license is preserved at the project root. The Zenodo metadata separately
labels the deposit CC-BY-4.0. Both source licensing signals are preserved in
`zenodo_record.json`; this project does not claim to relicense the data.
No upstream modeling code or supplied journal PDF is included.
The new benchmark code is distributed under AGPL-3.0-only.

## Splits (zero-based, stop-exclusive)

- Training pool [0,240): 10 days. Smaller budgets use the most recent
  72 or 120 observations ending at index 239, giving 71/119/239 pairs.
- Validation [240,288): 48 observations, 47 internal transitions.
- Test [288,385): 97 observations, 96 internal transitions.
- The two transitions crossing split boundaries are deliberately excluded.
- 24-hour test forecasts have 73 overlapping start times.

Smaller budgets differ in both sample count and covered operating conditions.
This is a recent-data-budget experiment, not a pure i.i.d. learning curve.
