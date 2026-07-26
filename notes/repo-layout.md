# Target layout

This is the DESTINATION, not the current state. We refactor into this
at session 5, once the model works. Do not build it in advance.

iowa-corn/
├── README.md
├── pyproject.toml
├── Makefile
├── config.yaml
├── .env.example
│
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
│       └── panel.parquet
│
├── src/corn/
│   ├── config.py
│   ├── paths.py
│   ├── schemas.py
│   ├── ingest/
│   │   ├── nass.py
│   │   ├── gee.py
│   │   └── weather.py
│   ├── features/
│   │   ├── phenology.py
│   │   └── panel.py
│   ├── models/
│   │   ├── splits.py
│   │   ├── train.py
│   │   └── evaluate.py
│   └── cli.py
│
├── gee/
│   └── s2_county_stats.py
│
├── tests/
│   ├── test_fips.py
│   ├── test_asof.py
│   ├── test_splits.py
│   └── test_schemas.py
│
├── notebooks/
└── reports/
