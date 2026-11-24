# BatteryX

BatteryX estimates lithium-ion battery state of health (SOH) and remaining useful life (RUL). Its trained model combines discharge-curve features, cycle history, operating conditions, and attention over nearby reference cells. The API returns raw and projected SOH, RUL in cycles, and conformal RUL bounds.

## Cell Lab and inference API

Open the Cell Lab at the service root. The approved interface contains an example cell, a custom cycle CSV reader, capacity history, and a results view. The browser calculates relative SOH from the first recorded capacity and extrapolates RUL to 80% of that reference for the interactive demo.

Use the interactive API docs at /docs to run the trained checkpoint through POST /api/predict. Supply at least 20 increasing cycle indices with aligned measured capacities, operating conditions, and discharge curves preprocessed with the training pipeline. Each curve has 64 normalized capacity and derivative pairs on a shared increasing voltage grid. See model/batteryx_core.py for cycle_summary and make_cell.

The Cell Lab's CSV workflow runs locally in the browser. The trained model endpoint requires the additional measured curve and operating-condition inputs.

## Evaluation

| Evaluation | SOH MAE | RUL MAE | Cell-level interval coverage |
| --- | ---: | ---: | ---: |
| Five-fold in-distribution, projected SOH | 0.0107 | 170.7 cycles | 90.5% |
| Leave-one-dataset-out transfer | 0.0277 | 448.1 cycles | — |

Figures come from the [v3.0 model card](results/model_card.md) for seeds 42, 43, and 44. Coverage is empirical, and a new dataset can have different coverage.

## Run locally

Use Python 3.12 in a virtual environment.

```sh
python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.7.1
python -m pip install -r requirements.txt
python -m uvicorn app:app --host 127.0.0.1 --port 8000
```

Open [Cell Lab](http://127.0.0.1:8000/) or [API docs](http://127.0.0.1:8000/docs). Check `/api/ready` to confirm the trained model loaded.

## Deploy on Render

Create a Blueprint from this repository and use the root `render.yaml`. It provisions a Python Web Service that serves the Cell Lab and API at one URL. The service checks `/api/ready` before reporting readiness.

## Repository layout

- `frontend/index.html` contains the approved Cell Lab.
- `app.py` serves the lab and inference API.
- `model/` contains model code, weights, configuration, normalization statistics, and the compact reference bank.
- `results/` contains the model card and selected evaluation tables.
