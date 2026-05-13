# Pure TCN Model for Hydrological Discharge Forecasting

## Project Overview

This project implements a **Pure Data-Driven Temporal Convolutional Network (TCN)** for river discharge prediction using only meteorological and hydrological variables (no physics-based model output like TOPKAPI `Q`).

**Dataset**: Casalecchio basin (hourly data from 2013 to 2026)

---

## Key Features

- Pure data-driven approach (no reliance on TOPKAPI predictions)
- Temporal Convolutional Network (TCN) with dilated convolutions
- 13 input features (rainfall, temperature, soil moisture, snow, etc.)
- 24-hour lookback window
- Trained with L2 regularization and early stopping

---

## Performance Results

| Model              | MAE (m³/s) | RMSE (m³/s) |
|--------------------|------------|-------------|
| Pure TCN (This Project) | **8.33**   | **17.23**   |
| TOPKAPI (Baseline) | 11.62      | 19.99       |

**Improvement**: ~28% better than TOPKAPI baseline.

---

Technologies Used

Python 3.13
PyTorch (CPU)
Pandas, NumPy, scikit-learn
Temporal Convolutional Network (TCN)




Author
Farhad Bayrami
Hydrological Discharge Forecasting using TCN
