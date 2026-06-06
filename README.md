# Four-Factor Resilience Against Deaths of Despair

Replication code and data for:

Four-Factor Resilience Against Deaths of Despair: Formal Analysis, Agent-Based Simulation, and Cross-National Evidence.

## Contents

- `code/` — Python replication script (produces all figures and tables)
- `data/` — Analytic sample and supplementary tables (N = 147 countries)
- `figures/` — All manuscript figures (300 DPI PNG)

## Requirements

```bash
pip install numpy pandas scipy statsmodels matplotlib networkx requests wbgapi
```

## Usage

```bash
python code/four_factor_resilience_final.py
```

## Data Sources

- WHO Global Health Estimates 2019 (suicide mortality)
- World Values Survey Wave 7 / Pew Research / Gallup (religiosity)
- World Bank WDI 2019 (GDP per capita, unemployment)
- World Happiness Report 2020 (social support)
