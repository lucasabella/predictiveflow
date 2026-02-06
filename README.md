# PredictiveFlow

An end-to-end predictive maintenance system that ingests industrial sensor data, predicts remaining useful life (RUL) of equipment, detects anomalies, serves predictions via a REST API, and visualizes everything on a live dashboard.

## Dataset

**NASA C-MAPSS Turbofan Engine Degradation Simulation** (FD001 subset for MVP)

- 100 training engines, 100 test engines
- 21 sensor channels + 3 operational settings per cycle
- Run-to-failure data (natural RUL labels)
- Source: [NASA Prognostics Data Repository](https://www.nasa.gov/content/prognostics-center-of-excellence-data-set-repository)

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.11+ |
| Data processing | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| ML — traditional | scikit-learn |
| ML — deep learning | PyTorch |
| API | FastAPI |
| Dashboard | Streamlit |
| Containerization | Docker + docker-compose |
| CI/CD | GitHub Actions |
| Testing | pytest |
| Linting | Ruff |

## Project Structure

```
predictiveflow/
├── data/
│   ├── raw/                    # Original C-MAPSS files
│   └── processed/              # Cleaned, feature-engineered data
├── docs/                       # Project documentation
├── notebooks/
│   ├── 01_exploration.ipynb    # EDA, sensor analysis
│   ├── 02_preprocessing.ipynb  # Feature engineering experiments
│   └── 03_modeling.ipynb       # Model comparison & evaluation
├── src/
│   ├── data/                   # Data loading, preprocessing, features
│   ├── models/                 # ML models (RF, Isolation Forest, LSTM)
│   ├── api/                    # FastAPI service
│   └── dashboard/              # Streamlit app
├── models/                     # Saved trained models (.joblib, .pt)
├── tests/                      # Unit + integration tests
├── requirements.txt
└── README.md
```

## Quick Start

### Prerequisites

- Python 3.11+

### Setup

```bash
# Clone the repo
git clone https://github.com/yourusername/predictiveflow.git
cd predictiveflow

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Run the Exploration Notebook

```bash
jupyter notebook notebooks/01_exploration.ipynb
```

This loads the FD001 dataset and walks through:
- Data loading and inspection
- RUL (Remaining Useful Life) computation
- Engine lifetime distribution
- Sensor variance analysis (identifying useless sensors)
- Sensor degradation traces over engine lifetime
- Sensor correlation with RUL
- RUL target distribution (raw vs. capped at 125)

## Documentation

See the [docs/](docs/) folder for detailed documentation:

- [Getting Started](docs/getting-started.md) — Setup, installation, and first steps
- [Changelog](docs/changelog.md) — History of major changes

## Current Status

**Phase 1 — Data Pipeline (in progress)**
- [x] Project structure created
- [x] Dataset loaded (FD001)
- [x] Exploration notebook (`01_exploration.ipynb`)
- [ ] Preprocessing pipeline (`src/data/`)
- [ ] Feature engineering
