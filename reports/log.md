# Build log

## 2026-07-26 - repo initialized
Nothing built yet. Target: beat the county+year trend baseline on
2025 Iowa corn yields using Sentinel-2 phenology.
Baseline RMSE: unknown.

## 2026-07-26 - labels + trend baseline (NASS pull 2026-07-26)
County-mean (floor) and county+trend baselines, year-blocked splits. Source: scripts/01_labels.py.

LOYO 2017-2024 (bu/ac RMSE):
  2017: floor=13.907  trend=19.702  n_unseen_counties=0
  2018: floor=15.515  trend=15.379  n_unseen_counties=0
  2019: floor=12.581  trend=12.392  n_unseen_counties=0
  2020: floor=29.243  trend=28.799  n_unseen_counties=0
  2021: floor=12.899  trend=12.410  n_unseen_counties=0
  2022: floor=13.791  trend=14.049  n_unseen_counties=0
  2023: floor=18.177  trend=17.321  n_unseen_counties=0
  2024: floor=21.995  trend=20.349  n_unseen_counties=0
  pooled: floor=18.067  trend=18.344

2025 holdout: floor=17.131  trend=12.060  n_unseen_counties=0

Notes:
- NASS pull date: 2026-07-26. NASS revises county estimates; 2025 figures may still move on a later pull.
- NASS suppresses a county-year when too few operations report, which correlates with low corn acreage. The labeled set skews toward higher-production counties, so every RMSE above is conditional on that subset.
- 2017 and 2024 are the two LOYO folds where the year term extrapolates outside the training range (2018-2024 and 2017-2023 respectively); 2018-2023 interpolate. The fitted year slope on the 8 possible 7-year folds ranges ~0.56 to ~2.86 bu/ac/yr, so extrapolation-fold error is sensitive to which years are excluded, not a bug. This is also why pooled trend (18.34) roughly matches pooled floor (18.07) despite trend beating floor in 6 of 8 individual folds.

## 2026-07-26 - long-run trend + anomaly labels (NASS pull 2026-07-26)
Three baselines, same year-blocked splits: floor (county mean), refit_trend (county + year re-estimated per fold, prior session's baseline), fixed_trend (county + year slope frozen from a 1995-2024 fit). Source: scripts/01_labels.py.

Long-run trend fit on 1995-2024 (n=2886): slope=2.3858 bu/ac/yr, SE=0.0400. This slope is frozen and used as-is for fixed_trend below -- not refit inside any fold.

LOYO 2017-2024 (bu/ac RMSE):
  2017: floor=13.907  refit_trend=19.702  fixed_trend=18.414  n_unseen_counties=0
  2018: floor=15.515  refit_trend=15.379  fixed_trend=15.723  n_unseen_counties=0
  2019: floor=12.581  refit_trend=12.392  fixed_trend=12.493  n_unseen_counties=0
  2020: floor=29.243  refit_trend=28.799  fixed_trend=28.483  n_unseen_counties=0
  2021: floor=12.899  refit_trend=12.410  fixed_trend=12.172  n_unseen_counties=0
  2022: floor=13.791  refit_trend=14.049  fixed_trend=14.366  n_unseen_counties=0
  2023: floor=18.177  refit_trend=17.321  fixed_trend=17.359  n_unseen_counties=0
  2024: floor=21.995  refit_trend=20.349  fixed_trend=16.022  n_unseen_counties=0

Summary (RMSE, bu/ac):
  baseline      pooled_loyo  interior_loyo  holdout_2025
  floor              18.067         18.126        17.131
  refit_trend        18.344         17.806        12.060
  fixed_trend        17.715         17.806        10.520
  (n_unseen_counties in 2025 holdout: 0)

Superseded: this entry originally wrote data/processed/yield_anomaly.parquet with a pre-computed yield_anomaly using county intercepts fit on all of 2017-2024, applied in-sample back onto those same rows -- that leaks each row's own value into its own target through the county mean. See the next entry for the fix.

Notes:
- NASS pull date: 2026-07-26. Label pull now spans 1995-2025 so the long-run trend has enough years to be stable; the modeling window (LOYO + holdout) is unchanged at 2017-2025. The trend slope is fit on 1995-2024 and never touches 2025.
- NASS suppresses a county-year when too few operations report, which correlates with low corn acreage. The labeled set skews toward higher-production counties, so every RMSE above is conditional on that subset.
- refit_trend re-estimates the year slope inside each 7-8 year fold, which let it absorb weather rather than technology (fold slopes ranged ~0.56-2.86 bu/ac/yr in the prior session). fixed_trend pins the slope at 2.386 (SE=0.040) from the 30-year fit and only refits county intercepts per fold.
- interior_loyo (2018-2023) excludes the two folds where the year term extrapolates (2017, 2024); compare it to pooled_loyo above to see whether trend's LOYO performance is an edge-fold artifact or holds throughout.

## 2026-07-26 - fixed-slope diagnostics + leakage-free labels (NASS pull 2026-07-26)
Three baselines, same year-blocked splits: floor (county mean), refit_trend (county + year re-estimated per fold), fixed_trend (county + year slope frozen from a 1995-2024 fit). Source: scripts/01_labels.py.

Long-run trend fit on 1995-2024 (n=2886): slope=2.3858 bu/ac/yr, SE=0.0400. This slope is frozen and used as-is for fixed_trend below -- not refit inside any fold.

LOYO 2017-2024 (bu/ac RMSE, plus that fold's own refit slope):
  2017: floor=13.907  refit_trend=19.702  fixed_trend=18.414  refit_slope=2.858  n_unseen_counties=0
  2018: floor=15.515  refit_trend=15.379  fixed_trend=15.723  refit_slope=1.700  n_unseen_counties=0
  2019: floor=12.581  refit_trend=12.392  fixed_trend=12.493  refit_slope=1.593  n_unseen_counties=0
  2020: floor=29.243  refit_trend=28.799  fixed_trend=28.483  refit_slope=1.370  n_unseen_counties=0
  2021: floor=12.899  refit_trend=12.410  fixed_trend=12.172  refit_slope=1.508  n_unseen_counties=0
  2022: floor=13.791  refit_trend=14.049  fixed_trend=14.366  refit_slope=1.700  n_unseen_counties=0
  2023: floor=18.177  refit_trend=17.321  fixed_trend=17.359  refit_slope=1.523  n_unseen_counties=0
  2024: floor=21.995  refit_trend=20.349  fixed_trend=16.022  refit_slope=0.557  n_unseen_counties=0

Summary (RMSE, bu/ac):
  baseline      pooled_loyo  interior_loyo  holdout_2025
  floor              18.067         18.126        17.131
  refit_trend        18.344         17.806        12.060
  fixed_trend        17.715         17.806        10.520
  (n_unseen_counties in 2025 holdout: 0)

Mean out-of-fold anomaly by year (yield - fixed_trend prediction, bu/ac; 2017-2024 from that year's LOYO fold, 2025 from the holdout -- never a year's own data in its own trend fit):
  2017: +12.748
  2018: +3.538
  2019: +1.672
  2020: -21.071
  2021: +4.248
  2022: -4.038
  2023: -1.717
  2024: +6.832
  2025: +2.854

Labels written to data/processed/labels.parquet: fips, year, yield_bu_ac, trend_slope_component (= frozen slope * year, no fold data involved). No anomaly column is pre-computed -- session 4 fits county intercepts from each fold's training rows and builds yield_anomaly = yield_bu_ac - trend_slope_component - county_intercept inside the fold.

Notes:
- NASS pull date: 2026-07-26. Label pull spans 1995-2025 so the long-run trend has enough years to be stable; the modeling window (LOYO + holdout) is unchanged at 2017-2025. The trend slope is fit on 1995-2024 and never touches 2025.
- NASS suppresses a county-year when too few operations report, which correlates with low corn acreage. The labeled set skews toward higher-production counties, so every RMSE above is conditional on that subset.
- refit_trend re-estimates the year slope inside each 7-8 year fold, which let it absorb weather rather than technology -- see the per-fold refit_slope column above for the actual spread. fixed_trend pins the slope at 2.386 (SE=0.040) from the 30-year fit and only refits county intercepts per fold.
- interior_loyo (2018-2023) excludes the two folds where the year term extrapolates (2017, 2024); compare it to pooled_loyo above to see whether trend's LOYO performance is an edge-fold artifact or holds throughout. Investigated a near-tie between refit_trend and fixed_trend on interior_loyo (both ~17.806): confirmed NOT a bug -- per-fold refit slopes for interior folds cluster tightly (~1.37-1.7 bu/ac/yr, since every interior training set still spans both 2017 and 2024) versus the fixed slope's 2.386, but the two pooled RMSEs differ starting at the 4th decimal (17.80635 vs 17.80610) -- a genuine coincidence in the aggregate, not identical predictions. The wide 0.56-2.86 slope range from the original run only appears in the two edge folds (2017, 2024), where the training set is missing an endpoint year.
- Removed data/processed/yield_anomaly.parquet from an earlier version of this script: it pre-computed yield_anomaly using county intercepts fit on all of 2017-2024, which let each row's own value leak into its own target through the county mean. See CLAUDE.md non-negotiables.
