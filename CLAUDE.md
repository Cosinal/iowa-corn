# Iowa corn yield model

## What this is
County-level corn yield prediction from Sentinel-2 phenology.
Train 2017–2024, holdout 2025. n ≈ 890 rows. This is a SMALL data project.

## Non-negotiables
- Never random-split. Splits are year-blocked only. See src/corn/models/splits.py.
- FIPS codes are strings, always. Assert 5 chars, 99 unique Iowa counties.
- In-season features must respect the as-of cutoff. A week-30 model sees
  only data through week 30. No exceptions.
- Every model result is reported against the trend baseline (county + year).
  A number without the baseline next to it is not a result.

## Anti-overengineering
Default to the boring option. Do NOT add without me asking:
orchestrators, MLflow, Docker, feature stores, ABCs, plugin systems,
config frameworks, retry decorators, async, or a web API.
LightGBM only — no neural nets. If a net beats LightGBM on 890 rows,
assume leakage and go looking for it.
Prefer one function over one class. Prefer a script over a framework.

## Data
Raw is immutable. Never edit in place. All heavy pixel work happens
server-side in Earth Engine; only county-week aggregates come local.

## Before you write code
State your plan first. If a task touches splits, features, or the
as-of cutoff, stop and confirm with me before implementing.