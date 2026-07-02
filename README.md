# 💻 Laptop Price Prediction — Regression Task

A machine learning project that predicts laptop prices using regression models trained on a real-world uncleaned dataset.

---

## 📁 Project Structure

```
├── ML_laptop_price.ipynb   # Main notebook
└── README.md
```

---

## 📋 Dataset

- **Source:** [Uncleaned Laptop Price Dataset](https://www.kaggle.com/datasets/ehtishamsadiq/uncleaned-laptop-price-dataset) — Kaggle
- **Size:** 1,303 rows × 12 columns
- **Target:** `Price` (continuous — regression task)

---

## 🔄 Project Pipeline

### 1. Data Loading & First Look
- Loaded dataset, inspected shape, column types, and first rows

### 2. Exploratory Data Analysis (EDA)
- Summary statistics for all features
- Feature distributions (histograms)
- Outlier detection (boxplot on `Inches`)

### 3. Data Cleaning
- Dropped fully null rows (30 rows)
- Removed duplicate entries
- Fixed `?` values in `Inches` and `Weight` → imputed with **median** (robust to outliers)
- Fixed data entry errors in `Inches` (e.g., 35.6 → 15.6)
- Standardized `Memory` units (TB → GB)

### 4. Feature Engineering
Extracted **14 new features** from raw columns:

| Feature | Source | Type |
|---------|--------|------|
| ScreenHeight / ScreenWidth | ScreenResolution | Numeric |
| IPS | ScreenResolution | Binary |
| TouchScreen | ScreenResolution | Binary |
| CpuBrand | Cpu | Categorical |
| CpuSeries | Cpu | Categorical |
| CpuGHz | Cpu | Numeric |
| CpuFamily | Cpu | Categorical |
| Gpu_Brand | Gpu | Categorical |
| GpuSeries | Gpu | Categorical |
| SSD / HDD / Flash / Hybrid | Memory | Binary |
| MemoryGB | Memory | Numeric |

### 5. Preprocessing
- **Encoding:** One-Hot Encoding for high-cardinality categoricals, Binary Encoding for `Company`
- **Scaling:** StandardScaler (required for Linear Regression)
- **Target transform:** `log(Price)` to fix right skew — applied **after** train/test split
- **PCA:** Applied at 95% variance threshold — reduced features but slightly hurt performance, so raw scaled features were used for best model

### 6. Modeling

| Model | R² Test | MAE | RMSE |
|-------|---------|-----|------|
| Linear Regression | ~0.78 | — | — |
| Random Forest | ~0.88 | — | — |
| **XGBoost (tuned)** | **0.899** | **0.142** | **0.189** |

> XGBoost hyperparameters tuned using `RandomizedSearchCV` with 5-fold cross-validation

### 7. Evaluation
- Metrics: R², MAE, MSE, RMSE
- Plots: Metrics bar chart, Actual vs Predicted, Learning Curve

---

## 🏆 Best Model — XGBoost

```
R² Train : 0.977
R² Test  : 0.899
MAE      : 0.142  (log scale)
RMSE     : 0.189  (log scale)
```

> Note: MAE and RMSE are in log scale. To convert back: `np.exp(0.142) ≈ 15% average error`

---

## ⚙️ How to Run

1. Open `ML_laptop_price.ipynb` in Google Colab
2. Run all cells in order (`Runtime → Run all`)
3. Dataset downloads automatically via `kagglehub`

---

## 🔮 Future Improvements

- Try other encoding strategies and feature selection methods
- Experiment with stronger regularization to reduce overfitting gap (~8%)
- Test ensemble stacking (XGBoost + Random Forest)
- Add more data to improve validation curve convergence
