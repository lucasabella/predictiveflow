# Getting Started

## Prerequisites

- Python 3.11 or higher
- Git

## Installation

1. **Clone the repository**

```bash
git clone https://github.com/yourusername/predictiveflow.git
cd predictiveflow
```

2. **Create a virtual environment**

```bash
python3 -m venv venv
```

3. **Activate the virtual environment**

```bash
# macOS / Linux
source venv/bin/activate

# Windows
venv\Scripts\activate
```

4. **Install dependencies**

```bash
pip install -r requirements.txt
```

## Dataset

The project uses the NASA C-MAPSS FD001 dataset. The raw data files should be placed in `data/raw/`:

```
data/raw/
├── train_FD001.txt
├── test_FD001.txt
├── RUL_FD001.txt
└── readme.txt
```

### Data format

The text files are space-delimited with no headers. The 26 columns are:

| Columns | Name |
|---|---|
| 0 | engine_id |
| 1 | cycle |
| 2-4 | op_setting_1, op_setting_2, op_setting_3 |
| 5-25 | sensor_1 through sensor_21 |

## Running the Notebooks

Start Jupyter from the project root:

```bash
jupyter notebook
```

Then open notebooks in order:

1. **`notebooks/01_exploration.ipynb`** — Loads the FD001 dataset and performs exploratory data analysis:
   - Loads train, test, and RUL files with proper column names
   - Computes Remaining Useful Life (RUL) for each row
   - Caps RUL at 125 cycles (piecewise linear)
   - Visualizes engine lifetime distribution
   - Identifies near-constant sensors (candidates for removal)
   - Plots sensor degradation traces over engine lifetime
   - Shows sensor correlation with RUL
   - Compares raw vs. capped RUL distributions

## Project Structure

```
predictiveflow/
├── data/raw/          # Original C-MAPSS files (not committed to git)
├── data/processed/    # Cleaned, feature-engineered data
├── docs/              # This documentation
├── notebooks/         # Jupyter notebooks for EDA and experiments
├── src/data/          # Data loading, preprocessing, feature engineering
├── src/models/        # ML model definitions and training
├── src/api/           # FastAPI REST service
├── src/dashboard/     # Streamlit dashboard
├── models/            # Saved trained models (.joblib, .pt)
├── tests/             # Unit and integration tests
└── requirements.txt   # Python dependencies
```
