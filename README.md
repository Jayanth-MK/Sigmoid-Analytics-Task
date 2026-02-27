# Customer Churn Prediction — CRISP-DM Pipeline

Binary classification pipeline following the CRISP-DM methodology.

## Setup
```bash
pip install -r requirements.txt
```

## Data

Place both files inside the `data/` folder:
- `train__6_.csv`
- `train_churn_labels.csv`

## Run
```bash
python main.py
```

## Outputs (auto-created in `outputs/`)

| File | Description |
|---|---|
| `crisp_dm_results.png` | 10-panel results visualization |
| `best_model.pkl` | Serialized best model |
| `model_metadata.json` | Metrics, threshold, feature list |
| `crisp_dm_report.txt` | Full text run log |
| `validation_predictions.csv` | Per-row predictions on val set |

## Project Structure
```
churn_prediction/
├── data/                    # input CSVs (not committed to git)
├── src/
│   ├── data_loader.py       # Phase 2 — load & profile data
│   ├── data_preparation.py  # Phase 3 — clean, encode, split
│   ├── feature_engineering.py # Phase 3 — extendable transforms
│   ├── modeling.py          # Phase 4 — train all models
│   ├── evaluation.py        # Phase 5 — metrics, threshold, save
│   └── visualization.py     # Phase 5 — 10-panel figure
├── outputs/                 # auto-created at runtime
├── config.py                # all hyperparameters and paths
├── main.py                  # entry point, orchestrates all phases
├── requirements.txt
└── README.md
```

## Extending the Pipeline

- **Add a model**: edit `get_model_definitions()` in `src/modeling.py`
- **Change hyperparameters**: edit `config.py` only
- **Add features**: edit `engineer_features()` in `src/feature_engineering.py`
- **Change thresholds or splits**: edit `config.py` only
```

---

## FILE 12: `.gitignore`
```
# Data (large files — do not commit)
data/

# Outputs
outputs/

# Python
__pycache__/
*.pyc
*.pyo
*.pyd
.Python
*.egg-info/
dist/
build/

# Virtual environments
venv/
env/
.venv/

# IDE
.vscode/
.idea/

# Jupyter
.ipynb_checkpoints/

# OS
.DS_Store
Thumbs.db