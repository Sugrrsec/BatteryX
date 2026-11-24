# Model Card: BatteryX Degradation Prognostics

## Model Details
- **Architecture**: BatteryX (CurveCNN + Transformer SeqEnc + CellGAT with 21-dim Outcome Provenance)
- **Framework**: PyTorch with ModelEMA (training AMP: True)
- **Version**: 3.0 (cycle-space auxiliary objective and fade-rate supervision)
- **Run Mode**: full (Seeds: [42, 43, 44])

## Intended Use
Cross-chemistry battery state-of-health estimation and remaining useful life forecasting.

## Performance Metrics (Computed Dynamically from this Run)
- **5-Fold In-Distribution SOH MAE (Monotone Projected)**: 0.0107
- **5-Fold In-Distribution RUL MAE**: 170.7 cycles
- **5-Fold Conformal Cell-Level Coverage**: 90.5%
- **Leave-One-Dataset-Out (LODO) SOH MAE**: 0.0277
- **Leave-One-Dataset-Out (LODO) RUL MAE**: 448.1 cycles

## Scientific Limitations & Disclosures
- Capacity-derived SOH predictions rely on capacity measurements and are labeled trivial proxies.
- Cross-domain conformal coverage represents empirical transfer coverage without formal exchangeability guarantees.