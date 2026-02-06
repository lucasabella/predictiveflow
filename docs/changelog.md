# Changelog

All major changes to PredictiveFlow are documented here.

---

## 2026-02-06 — Project Initialization

### Added
- Project directory structure (`data/`, `src/`, `notebooks/`, `tests/`, `models/`, `docs/`)
- `requirements.txt` with initial dependencies (pandas, numpy, matplotlib, seaborn, scikit-learn, jupyter)
- `.gitignore` configured for Python and JetBrains IDE files
- `notebooks/01_exploration.ipynb` — FD001 exploratory data analysis notebook
  - Loads train/test/RUL data with proper column names
  - Computes and caps RUL at 125 cycles
  - Engine lifetime distribution histogram
  - Sensor variance analysis (identifies near-constant sensors)
  - Sensor degradation traces for sample engines
  - Sensor-RUL correlation chart
  - Raw vs. capped RUL distribution comparison
- `README.md` with project overview, tech stack, setup instructions, and current status
- `docs/getting-started.md` — detailed setup and usage guide