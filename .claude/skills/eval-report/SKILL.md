---
name: eval-report
description: Run the full evaluation protocol and write a results report. Use whenever the user asks to evaluate, score, benchmark, or "see how the model did", or after any change to features or model code.
---

# Evaluation protocol

Never report a model number in isolation. Run all four steps, in order.

## 1. Trend baseline
Fit yield ~ county fixed effect + linear year. This is the bar.
Compute its RMSE under the same splits as the model.

## 2. Leave-one-year-out
For each year 2017-2024: train on all other years, predict that year.
Report per-year RMSE and the mean. Per-year spread matters more than
the mean — a model that's great except in drought years is not a model.

## 3. 2025 holdout
Train on 2017-2024 only. Predict 2025. Single number, no tuning
against it. If you tuned anything after seeing this, say so explicitly.

## 4. Skill by week
Retrain with as-of truncation at weeks 24, 28, 32, 36.
Plot RMSE vs week. Skill must improve monotonically through the season;
if it doesn't, suspect leakage.

## Output
Write to reports/YYYY-MM-DD_eval.md with a table: split | baseline RMSE |
model RMSE | delta. Then a short "what I'd distrust about this" section.
Append a one-line summary to reports/log.md.