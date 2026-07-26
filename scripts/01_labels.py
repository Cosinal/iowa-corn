"""Pull Iowa county corn GRAIN yield from NASS QuickStats and fit the trend baseline.

The label pull covers 1995-2025 so the long-run trend has enough years to be
stable, but the modeling window (LOYO + holdout) stays 2017-2025 -- that's the
period src/corn will actually train and evaluate on.

Baselines fit here (all under year-blocked splits, never random):
  - floor:        county mean, no year term
  - refit_trend:  county fixed effect + year term re-estimated inside each fold
  - fixed_trend:  county fixed effect + year slope frozen from the 1995-2024 fit

refit_trend re-estimates the year slope on only 7-8 years of data, which lets
it absorb weather instead of technology (slopes ranged 0.56-2.86 bu/ac/yr
across folds in the prior run). fixed_trend pins the slope from the 30-year
fit and only refits the county intercepts per fold, so it can't do that.

Evaluated three ways:
  - leave-one-year-out across 2017-2024 (per-year RMSE, pooled RMSE)
  - interior-only LOYO (2018-2023), where the year term interpolates instead
    of extrapolating -- checks whether trend's LOYO performance is an
    edge-fold artifact
  - 2025 holdout, trained on all of 2017-2024

Also writes data/processed/labels.parquet: fips, year, yield_bu_ac, and
trend_slope_component (slope * year -- the frozen-slope term only, using no
fold-specific data). It deliberately does NOT include a pre-computed anomaly:
county intercepts must come from a fold's training data only, so session 4
builds yield_anomaly = yield_bu_ac - trend_slope_component - county_intercept
inside each fold, using that fold's own training-fold intercepts. Baking the
anomaly in here would fit the intercept on a year's own held-out data and
leak it into its own target.
"""
import os
from datetime import date

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
from sklearn.linear_model import LinearRegression

load_dotenv()

NASS_URL = "https://quickstats.nass.usda.gov/api/api_GET/"
YEAR_START = 1995
YEAR_END = 2025
LONGRUN_YEARS = range(1995, 2025)  # 1995-2024, for freezing the year slope
MODEL_YEAR_START = 2017
TRAIN_YEARS = range(2017, 2025)  # 2017-2024
INTERIOR_YEARS = range(2018, 2024)  # 2018-2023, year term interpolates
HOLDOUT_YEAR = 2025
PULL_DATE = date.today().isoformat()

RAW_PATH = "data/raw/nass_yields.parquet"
LABELS_PATH = "data/processed/labels.parquet"
LOG_PATH = "reports/log.md"

KNOWN_AGGREGATE_NAMES = {"OTHER COUNTIES", "OTHER (COMBINED) COUNTIES"}


def pull_nass_yields() -> pd.DataFrame:
    key = os.environ["NASS_API_KEY"]
    params = {
        "key": key,
        "commodity_desc": "CORN",
        "statisticcat_desc": "YIELD",
        "unit_desc": "BU / ACRE",
        "util_practice_desc": "GRAIN",
        "agg_level_desc": "COUNTY",
        "state_alpha": "IA",
        "source_desc": "SURVEY",
        "year__GE": str(YEAR_START),
        "year__LE": str(YEAR_END),
        "format": "JSON",
    }
    resp = requests.get(NASS_URL, params=params, timeout=60)
    resp.raise_for_status()
    df = pd.DataFrame(resp.json()["data"])
    print(f"NASS returned {len(df)} rows")

    is_real_county = df["county_ansi"] != ""
    dropped = df.loc[~is_real_county]
    print(f"Dropping {len(dropped)} non-county aggregate rows:")
    print(dropped[["year", "county_name"]].to_string(index=False))
    unexpected = set(dropped["county_name"].unique()) - KNOWN_AGGREGATE_NAMES
    assert not unexpected, f"unexpected non-county row types, inspect before dropping: {unexpected}"

    df = df.loc[is_real_county].copy()

    df["fips"] = df["state_ansi"].str.zfill(2) + df["county_ansi"].str.zfill(3)
    df["yield_bu_ac"] = pd.to_numeric(df["Value"], errors="raise")
    df["year"] = df["year"].astype(int)
    df["pull_date"] = PULL_DATE

    out = df[["fips", "county_name", "year", "yield_bu_ac", "pull_date"]].reset_index(drop=True)

    assert out["fips"].str.len().eq(5).all(), "FIPS codes must be 5 characters"
    n_counties = out["fips"].nunique()
    assert n_counties == 99, f"expected 99 unique Iowa counties (union across years), got {n_counties}"

    n_2025 = out.loc[out["year"] == HOLDOUT_YEAR, "fips"].nunique()
    assert n_2025 >= 50, (
        f"{HOLDOUT_YEAR} holdout year has implausibly few counties ({n_2025}) "
        "-- NASS may not have published it yet"
    )

    assert not out.duplicated(subset=["fips", "year"]).any(), (
        "duplicate (fips, year) rows -- check source_desc, a CENSUS row may have slipped in "
        "alongside SURVEY for a census year (2017, 2022)"
    )

    return out


def save_raw(df: pd.DataFrame) -> None:
    os.makedirs(os.path.dirname(RAW_PATH), exist_ok=True)
    df.to_parquet(RAW_PATH, index=False)
    print(f"Saved {len(df)} rows to {RAW_PATH}")


def rmse(actual, pred) -> float:
    actual = np.asarray(actual, dtype=float)
    pred = np.asarray(pred, dtype=float)
    return float(np.sqrt(np.mean((actual - pred) ** 2)))


def fit_predict_floor(train_df: pd.DataFrame, test_df: pd.DataFrame):
    county_means = train_df.groupby("fips")["yield_bu_ac"].mean()
    overall_mean = train_df["yield_bu_ac"].mean()
    preds = test_df["fips"].map(county_means)
    n_unseen = int(preds.isna().sum())
    preds = preds.fillna(overall_mean).to_numpy(dtype=float)
    return preds, n_unseen


def build_design(df: pd.DataFrame, all_counties: list) -> np.ndarray:
    fips_cat = pd.Categorical(df["fips"], categories=all_counties)
    county_dummies = pd.get_dummies(fips_cat, drop_first=True)
    X = pd.concat(
        [df[["year"]].reset_index(drop=True), county_dummies.reset_index(drop=True)], axis=1
    )
    return X.to_numpy(dtype=float)


def fit_predict_trend(train_df: pd.DataFrame, test_df: pd.DataFrame, all_counties: list):
    X_train = build_design(train_df, all_counties)
    y_train = train_df["yield_bu_ac"].to_numpy(dtype=float)
    model = LinearRegression()
    model.fit(X_train, y_train)

    X_test = build_design(test_df, all_counties)
    preds = model.predict(X_test)

    train_counties = set(train_df["fips"])
    unseen_mask = ~test_df["fips"].isin(train_counties).to_numpy()
    n_unseen = int(unseen_mask.sum())
    if n_unseen:
        # a county with zero training rows has an unidentifiable dummy coefficient;
        # sklearn silently sets it to 0 rather than warning, so override explicitly.
        preds = preds.copy()
        preds[unseen_mask] = y_train.mean()

    return preds, n_unseen, float(model.coef_[0])


def fit_longrun_trend(df: pd.DataFrame, all_counties: list):
    """OLS: yield ~ intercept + year + county dummies, fit on LONGRUN_YEARS.

    Returns (slope, slope_se) for the year term. Fit with an explicit design
    matrix (rather than sklearn) so the standard error is available via the
    closed-form OLS covariance, without adding a statsmodels dependency.
    """
    sub = df[df["year"].isin(LONGRUN_YEARS)]
    fips_cat = pd.Categorical(sub["fips"], categories=all_counties)
    dummies = pd.get_dummies(fips_cat, drop_first=True).to_numpy(dtype=float)
    year = sub["year"].to_numpy(dtype=float).reshape(-1, 1)
    intercept = np.ones((len(sub), 1))
    X = np.hstack([intercept, year, dummies])
    y = sub["yield_bu_ac"].to_numpy(dtype=float)

    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ X.T @ y
    resid = y - X @ beta
    dof = X.shape[0] - X.shape[1]
    sigma2 = np.sum(resid ** 2) / dof
    se = np.sqrt(np.diag(sigma2 * XtX_inv))

    slope, slope_se = float(beta[1]), float(se[1])
    print(
        f"Long-run trend ({LONGRUN_YEARS.start}-{LONGRUN_YEARS.stop - 1}): "
        f"slope={slope:.4f} bu/ac/yr  SE={slope_se:.4f}  n={len(sub)}"
    )
    return slope, slope_se


def fit_predict_fixed_trend(train_df: pd.DataFrame, test_df: pd.DataFrame, slope: float):
    """County fixed effect + year term pinned to `slope` (not re-estimated)."""
    resid = train_df["yield_bu_ac"] - slope * train_df["year"]
    county_effect = resid.groupby(train_df["fips"]).mean()
    overall_effect = resid.mean()

    test_effect = test_df["fips"].map(county_effect)
    n_unseen = int(test_effect.isna().sum())
    test_effect = test_effect.fillna(overall_effect).to_numpy(dtype=float)

    preds = test_effect + slope * test_df["year"].to_numpy(dtype=float)
    return preds, n_unseen


def append_log(
    slope, slope_se, loyo_rows, pooled, interior, holdout, n_unseen_holdout, n_longrun,
    mean_anomaly_by_year,
):
    lines = [
        "",
        f"## {date.today().isoformat()} - fixed-slope diagnostics + leakage-free labels (NASS pull {PULL_DATE})",
        "Three baselines, same year-blocked splits: floor (county mean), "
        "refit_trend (county + year re-estimated per fold), "
        "fixed_trend (county + year slope frozen from a 1995-2024 fit). "
        "Source: scripts/01_labels.py.",
        "",
        f"Long-run trend fit on {LONGRUN_YEARS.start}-{LONGRUN_YEARS.stop - 1} (n={n_longrun}): "
        f"slope={slope:.4f} bu/ac/yr, SE={slope_se:.4f}. This slope is frozen and used as-is "
        "for fixed_trend below -- not refit inside any fold.",
        "",
        "LOYO 2017-2024 (bu/ac RMSE, plus that fold's own refit slope):",
    ]
    for year, floor_r, refit_r, fixed_r, refit_slope, n_unseen in loyo_rows:
        lines.append(
            f"  {year}: floor={floor_r:.3f}  refit_trend={refit_r:.3f}  "
            f"fixed_trend={fixed_r:.3f}  refit_slope={refit_slope:.3f}  n_unseen_counties={n_unseen}"
        )
    lines.append("")
    lines.append("Summary (RMSE, bu/ac):")
    lines.append(f"  {'baseline':<12} {'pooled_loyo':>12} {'interior_loyo':>14} {'holdout_2025':>13}")
    for name in ("floor", "refit_trend", "fixed_trend"):
        lines.append(
            f"  {name:<12} {pooled[name]:>12.3f} {interior[name]:>14.3f} {holdout[name]:>13.3f}"
        )
    lines.append(f"  (n_unseen_counties in 2025 holdout: {n_unseen_holdout})")
    lines.append("")
    lines.append("Mean out-of-fold anomaly by year (yield - fixed_trend prediction, bu/ac; "
                  "2017-2024 from that year's LOYO fold, 2025 from the holdout -- never a "
                  "year's own data in its own trend fit):")
    for year in sorted(mean_anomaly_by_year):
        lines.append(f"  {year}: {mean_anomaly_by_year[year]:+.3f}")
    lines.append("")
    lines.append(
        f"Labels written to {LABELS_PATH}: fips, year, yield_bu_ac, trend_slope_component "
        "(= frozen slope * year, no fold data involved). No anomaly column is pre-computed -- "
        "session 4 fits county intercepts from each fold's training rows and builds "
        "yield_anomaly = yield_bu_ac - trend_slope_component - county_intercept inside the fold."
    )
    lines.append("")
    lines.append("Notes:")
    lines.append(
        f"- NASS pull date: {PULL_DATE}. Label pull spans 1995-2025 so the long-run trend has "
        "enough years to be stable; the modeling window (LOYO + holdout) is unchanged at "
        "2017-2025. The trend slope is fit on 1995-2024 and never touches 2025."
    )
    lines.append(
        "- NASS suppresses a county-year when too few operations report, which correlates "
        "with low corn acreage. The labeled set skews toward higher-production counties, "
        "so every RMSE above is conditional on that subset."
    )
    lines.append(
        "- refit_trend re-estimates the year slope inside each 7-8 year fold, which let it "
        "absorb weather rather than technology -- see the per-fold refit_slope column above "
        f"for the actual spread. fixed_trend pins the slope at {slope:.3f} (SE={slope_se:.3f}) "
        "from the 30-year fit and only refits county intercepts per fold."
    )
    lines.append(
        "- interior_loyo (2018-2023) excludes the two folds where the year term extrapolates "
        "(2017, 2024); compare it to pooled_loyo above to see whether trend's LOYO performance "
        "is an edge-fold artifact or holds throughout. Investigated a near-tie between "
        "refit_trend and fixed_trend on interior_loyo (both ~17.806): confirmed NOT a bug -- "
        "per-fold refit slopes for interior folds cluster tightly (~1.37-1.7 bu/ac/yr, since "
        "every interior training set still spans both 2017 and 2024) versus the fixed slope's "
        "2.386, but the two pooled RMSEs differ starting at the 4th decimal (17.80635 vs "
        "17.80610) -- a genuine coincidence in the aggregate, not identical predictions. The "
        "wide 0.56-2.86 slope range from the original run only appears in the two edge folds "
        "(2017, 2024), where the training set is missing an endpoint year."
    )
    lines.append(
        "- Removed data/processed/yield_anomaly.parquet from an earlier version of this script: "
        "it pre-computed yield_anomaly using county intercepts fit on all of 2017-2024, which "
        "let each row's own value leak into its own target through the county mean. See "
        "CLAUDE.md non-negotiables."
    )
    lines.append("")

    with open(LOG_PATH, "a") as f:
        f.write("\n".join(lines))
    print(f"Appended baseline results to {LOG_PATH}")


def main():
    df = pull_nass_yields()
    save_raw(df)

    all_counties = sorted(df["fips"].unique())
    n_longrun = int(df["year"].isin(LONGRUN_YEARS).sum())
    slope, slope_se = fit_longrun_trend(df, all_counties)

    model_df = df[df["year"].between(MODEL_YEAR_START, HOLDOUT_YEAR)].copy()
    train_pool = model_df[model_df["year"].isin(TRAIN_YEARS)].copy()

    print("\n--- Leave-one-year-out, 2017-2024 ---")
    print(
        f"{'year':>6}  {'floor':>8}  {'refit_trend':>12}  {'fixed_trend':>12}  "
        f"{'refit_slope':>11}  {'n_unseen':>8}"
    )
    loyo_rows = []
    floor_all, refit_all, fixed_all, actuals_all = [], [], [], []
    interior_floor, interior_refit, interior_fixed, interior_actuals = [], [], [], []
    mean_anomaly_by_year = {}

    for held_out in TRAIN_YEARS:
        train = train_pool[train_pool["year"] != held_out]
        test = train_pool[train_pool["year"] == held_out]
        actuals = test["yield_bu_ac"].to_numpy(dtype=float)

        floor_preds, n_unseen_floor = fit_predict_floor(train, test)
        refit_preds, n_unseen_refit, refit_slope = fit_predict_trend(train, test, all_counties)
        fixed_preds, n_unseen_fixed = fit_predict_fixed_trend(train, test, slope)
        assert n_unseen_floor == n_unseen_refit == n_unseen_fixed, (
            "baselines disagree on which test counties are unseen in training"
        )

        floor_r = rmse(actuals, floor_preds)
        refit_r = rmse(actuals, refit_preds)
        fixed_r = rmse(actuals, fixed_preds)
        print(
            f"{held_out:>6}  {floor_r:>8.3f}  {refit_r:>12.3f}  {fixed_r:>12.3f}  "
            f"{refit_slope:>11.3f}  {n_unseen_refit:>8}"
        )

        loyo_rows.append((held_out, floor_r, refit_r, fixed_r, refit_slope, n_unseen_refit))
        floor_all.append(floor_preds)
        refit_all.append(refit_preds)
        fixed_all.append(fixed_preds)
        actuals_all.append(actuals)
        mean_anomaly_by_year[held_out] = float(np.mean(actuals - fixed_preds))

        if held_out in INTERIOR_YEARS:
            interior_floor.append(floor_preds)
            interior_refit.append(refit_preds)
            interior_fixed.append(fixed_preds)
            interior_actuals.append(actuals)

    pooled = {
        "floor": rmse(np.concatenate(actuals_all), np.concatenate(floor_all)),
        "refit_trend": rmse(np.concatenate(actuals_all), np.concatenate(refit_all)),
        "fixed_trend": rmse(np.concatenate(actuals_all), np.concatenate(fixed_all)),
    }
    interior = {
        "floor": rmse(np.concatenate(interior_actuals), np.concatenate(interior_floor)),
        "refit_trend": rmse(np.concatenate(interior_actuals), np.concatenate(interior_refit)),
        "fixed_trend": rmse(np.concatenate(interior_actuals), np.concatenate(interior_fixed)),
    }

    print("\n--- 2025 holdout ---")
    test_2025 = model_df[model_df["year"] == HOLDOUT_YEAR]
    actuals_2025 = test_2025["yield_bu_ac"].to_numpy(dtype=float)

    floor_2025, n_unseen_floor_2025 = fit_predict_floor(train_pool, test_2025)
    refit_2025, n_unseen_refit_2025, refit_slope_2025 = fit_predict_trend(train_pool, test_2025, all_counties)
    fixed_2025, n_unseen_fixed_2025 = fit_predict_fixed_trend(train_pool, test_2025, slope)
    assert n_unseen_floor_2025 == n_unseen_refit_2025 == n_unseen_fixed_2025, (
        "baselines disagree on which 2025 counties are unseen in training"
    )

    holdout = {
        "floor": rmse(actuals_2025, floor_2025),
        "refit_trend": rmse(actuals_2025, refit_2025),
        "fixed_trend": rmse(actuals_2025, fixed_2025),
    }
    print(
        f"floor={holdout['floor']:.3f}  refit_trend={holdout['refit_trend']:.3f}  "
        f"fixed_trend={holdout['fixed_trend']:.3f}  refit_slope={refit_slope_2025:.3f}  "
        f"n_unseen_counties={n_unseen_fixed_2025}"
    )
    mean_anomaly_by_year[HOLDOUT_YEAR] = float(np.mean(actuals_2025 - fixed_2025))

    print("\n--- Summary (RMSE, bu/ac) ---")
    print(f"{'baseline':<12} {'pooled_loyo':>12} {'interior_loyo':>14} {'holdout_2025':>13}")
    for name in ("floor", "refit_trend", "fixed_trend"):
        print(f"{name:<12} {pooled[name]:>12.3f} {interior[name]:>14.3f} {holdout[name]:>13.3f}")

    print("\n--- Mean out-of-fold anomaly by year (yield - fixed_trend prediction, bu/ac) ---")
    for year in sorted(mean_anomaly_by_year):
        print(f"{year}: {mean_anomaly_by_year[year]:+.3f}")

    # data/processed/labels.parquet carries only the frozen-slope trend term,
    # which uses no fold data -- county intercepts (and therefore the anomaly)
    # get built inside each fold downstream, from that fold's training rows only.
    labels_df = model_df[["fips", "year", "yield_bu_ac"]].copy()
    labels_df["trend_slope_component"] = slope * labels_df["year"]
    labels_df = labels_df.sort_values(["fips", "year"]).reset_index(drop=True)
    os.makedirs(os.path.dirname(LABELS_PATH), exist_ok=True)
    labels_df.to_parquet(LABELS_PATH, index=False)
    print(f"\nSaved {len(labels_df)} rows to {LABELS_PATH}")

    append_log(
        slope, slope_se, loyo_rows, pooled, interior, holdout, n_unseen_fixed_2025, n_longrun,
        mean_anomaly_by_year,
    )


if __name__ == "__main__":
    main()
