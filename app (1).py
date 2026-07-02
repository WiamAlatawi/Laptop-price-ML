"""
Laptop Price Prediction — Interactive Streamlit App
====================================================
Deploys the full ML workflow from ML_laptop_price.ipynb as a web app:
1. Data Overview  2. EDA  3. Cleaning & Preprocessing  4. Model Results  5. Live Prediction

Run locally:   streamlit run app.py
Dataset:       laptopData.csv (place next to app.py, or it downloads via kagglehub)
"""

import os
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
from category_encoders import BinaryEncoder

# ----------------------------------------------------------------------------
# Page config & styling
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Laptop Price Predictor",
    page_icon=":computer:",
    layout="wide",
)

st.markdown(
    """
    <style>
      .main-title {font-size: 2.2rem; font-weight: 800; margin-bottom: 0;}
      .subtitle {color: #8a8f98; font-size: 1.05rem; margin-top: 0.2rem;}
      .metric-card {
          background: linear-gradient(135deg, rgba(99,102,241,0.10), rgba(99,102,241,0.03));
          border: 1px solid rgba(99,102,241,0.25);
          border-radius: 14px; padding: 1rem 1.2rem; text-align: center;
      }
      .metric-card h3 {margin: 0; font-size: 1.6rem;}
      .metric-card p  {margin: 0; color: #8a8f98; font-size: 0.85rem;}
      .price-box {
          background: linear-gradient(135deg, rgba(16,185,129,0.15), rgba(16,185,129,0.04));
          border: 1px solid rgba(16,185,129,0.4);
          border-radius: 16px; padding: 1.4rem; text-align: center;
      }
      .price-box h2 {margin: 0; font-size: 2.4rem; color: #10b981;}
      div[data-testid="stSidebar"] .stRadio label {font-size: 1.02rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

sns.set_style("whitegrid")
plt.rcParams["figure.autolayout"] = True


# ----------------------------------------------------------------------------
# 0. Data loading
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading dataset...")
def load_raw() -> pd.DataFrame:
    """Load laptopData.csv from local folder, or download from Kaggle."""
    if os.path.exists("laptopData.csv"):
        return pd.read_csv("laptopData.csv")
    try:
        import kagglehub
        path = kagglehub.dataset_download("ehtishamsadiq/uncleaned-laptop-price-dataset")
        return pd.read_csv(os.path.join(path, "laptopData.csv"))
    except Exception:
        return None


# ----------------------------------------------------------------------------
# Feature-engineering helpers (identical logic to the notebook)
# ----------------------------------------------------------------------------
def cpu_family(cpu: str) -> str:
    cpu = cpu.lower()
    if "core i3" in cpu: return "Core i3"
    if "core i5" in cpu: return "Core i5"
    if "core i7" in cpu: return "Core i7"
    if "core i9" in cpu: return "Core i9"
    if "xeon" in cpu: return "Xeon"
    if "ryzen" in cpu: return "Ryzen"
    if "celeron" in cpu: return "Celeron"
    if "pentium" in cpu: return "Pentium"
    if "atom" in cpu: return "Atom"
    if "a12" in cpu: return "A12"
    if "a10" in cpu: return "A10"
    if "a9" in cpu: return "A9"
    if "a8" in cpu: return "A8"
    if "a6" in cpu: return "A6"
    if "a4" in cpu: return "A4"
    if "e-series" in cpu: return "E-Series"
    if "fx" in cpu: return "FX"
    if "cortex" in cpu: return "Cortex"
    return "Other"


def gpu_series(gpu: str) -> str:
    gpu = gpu.lower()
    if "rtx" in gpu: return "RTX"
    if "gtx" in gpu: return "GTX"
    if "mx" in gpu: return "MX"
    if "quadro" in gpu: return "Quadro"
    if "rx" in gpu: return "RX"
    if "radeon" in gpu: return "Radeon"
    if "firepro" in gpu: return "FirePro"
    if "iris plus" in gpu: return "Iris Plus"
    if "iris" in gpu: return "Iris"
    if "uhd" in gpu: return "UHD"
    if "hd" in gpu: return "HD"
    if "graphics" in gpu: return "Graphics"
    if "mali" in gpu: return "Mali"
    return "Other"


@st.cache_data(show_spinner="Cleaning & engineering features...")
def clean_and_engineer(raw: pd.DataFrame):
    """Full cleaning + feature engineering pipeline from the notebook.
    Returns the processed dataframe plus before/after stats for the summary page."""
    df = raw.copy()
    before = {
        "shape": raw.shape,
        "missing": int(raw.isnull().sum().sum()),
        "duplicates": int(raw.duplicated().sum()),
        "n_cols": raw.shape[1],
    }

    # --- Cleaning ---
    df = df.drop(columns="Unnamed: 0", errors="ignore")   # row-number column
    df = df.dropna(axis=0)                                 # fully-NaN rows
    df = df.drop_duplicates()                              # duplicated rows

    # --- Inches: '?' -> NaN -> median, fix data-entry errors (25.6 -> 15.6 ...) ---
    df["Inches"] = df["Inches"].replace("?", np.nan).astype(float)
    df["Inches"] = df["Inches"].fillna(df["Inches"].median())
    inches_before_fix = df["Inches"].copy()
    fixes = {25.6: 15.6, 35.6: 15.6, 27.3: 17.3, 24: 14, 35.5: 15.5, 31.6: 11.6, 33.5: 13.5}
    df["Inches"] = df["Inches"].replace(fixes)

    # --- ScreenResolution -> ScreenWidth, ScreenHeight, IPS, TouchScreen ---
    hxw = df["ScreenResolution"].str.split().str[-1]
    df["ScreenHeight"] = hxw.str.split("x").str[-1].astype(int)
    df["ScreenWidth"] = hxw.str.split("x").str[0].astype(int)
    df["IPS"] = df["ScreenResolution"].str.contains("IPS").astype(int)
    df["TouchScreen"] = df["ScreenResolution"].str.contains("Touchscreen").astype(int)

    # --- Ram: remove 'GB' ---
    df["Ram"] = df["Ram"].str.replace("GB", "").astype(int)

    # --- Cpu -> brand, series, GHz, family ---
    df["CpuBrand"] = df["Cpu"].str.split().str[0]
    df["CpuSeries"] = df["Cpu"].str.split().apply(lambda x: " ".join(x[1:-1]))
    df["CpuGHz"] = df["Cpu"].str.split().str[-1].str.replace("GHz", "").astype(float)
    df["CpuFamily"] = df["CpuSeries"].apply(cpu_family)

    # --- Weight: '?' -> NaN, remove 'kg', median impute ---
    df["Weight"] = df["Weight"].replace("?", np.nan)
    df["Weight"] = df["Weight"].str.replace("kg", "").astype(float)
    df["Weight"] = df["Weight"].fillna(df["Weight"].median())

    # --- Memory: unify units, mode impute, storage-type flags + total GB ---
    df["Memory"] = df["Memory"].str.replace("1TB", "1024GB").str.replace("2TB", "2048GB")
    df["Memory"] = df["Memory"].str.replace("1.0TB", "1024GB")
    df["Memory"] = df["Memory"].replace("?", np.nan)
    df["Memory"] = df["Memory"].fillna(df["Memory"].mode()[0])
    df["SSD"] = df["Memory"].str.contains("SSD").astype(int)
    df["HDD"] = df["Memory"].str.contains("HDD").astype(int)
    df["Flash_Storage"] = df["Memory"].str.contains("Flash Storage").astype(int)
    df["Hybrid"] = df["Memory"].str.contains("Hybrid").astype(int)
    mem = df["Memory"].str.replace(r"SSD|HDD|Hybrid|Flash Storage|GB", "", regex=True)
    df["MemoryGB"] = mem.str.strip().str.split("+").apply(lambda parts: sum(int(p) for p in parts))

    # --- Gpu -> brand + series ---
    df["Gpu_Brand"] = df["Gpu"].str.split().str[0]
    df["GpuSeries"] = df["Gpu"].apply(gpu_series)

    # keep categorical options for the prediction page BEFORE encoding
    options = {
        "Company": sorted(df["Company"].unique().tolist()),
        "TypeName": sorted(df["TypeName"].unique().tolist()),
        "OpSys": sorted(df["OpSys"].unique().tolist()),
        "CpuBrand": sorted(df["CpuBrand"].unique().tolist()),
        "CpuFamily": sorted(df["CpuFamily"].unique().tolist()),
        "Gpu_Brand": sorted(df["Gpu_Brand"].unique().tolist()),
        "GpuSeries": sorted(df["GpuSeries"].unique().tolist()),
        "Ram": sorted(df["Ram"].unique().tolist()),
        "Resolutions": (
            df[["ScreenWidth", "ScreenHeight"]]
            .drop_duplicates()
            .sort_values("ScreenWidth")
            .apply(lambda r: f"{r.ScreenWidth} x {r.ScreenHeight}", axis=1)
            .tolist()
        ),
    }
    eda_df = df.copy()  # readable copy (pre one-hot) for EDA plots

    # --- One-hot encoding + drop raw text columns ---
    df = pd.get_dummies(
        df,
        columns=["TypeName", "OpSys", "CpuBrand", "Gpu_Brand", "CpuFamily", "GpuSeries"],
        dtype=int,
    )
    df = df.drop(columns=["ScreenResolution", "Cpu", "Memory", "Gpu", "CpuSeries"])

    after = {
        "shape": df.shape,
        "missing": int(df.isnull().sum().sum()),
        "duplicates": int(df.duplicated().sum()),
        "n_cols": df.shape[1],
    }
    return df, eda_df, options, before, after, inches_before_fix


@st.cache_resource(show_spinner="Training models (first run only)...")
def train_models(df: pd.DataFrame):
    """Split, encode Company (BinaryEncoder), scale, train the 3 notebook models."""
    X = df.drop(columns="Price")
    y = df["Price"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )
    # log-transform to fix the right-skewed Price
    y_train, y_test = np.log(y_train), np.log(y_test)

    encoder = BinaryEncoder(cols=["Company"])
    X_train_enc = encoder.fit_transform(X_train)
    X_test_enc = encoder.transform(X_test)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_enc)
    X_test_scaled = scaler.transform(X_test_enc)

    results = {}

    # 1) Linear Regression on PCA components (95% variance)
    pca = PCA(n_components=0.95)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    lin = LinearRegression().fit(X_train_pca, y_train)
    lp = lin.predict(X_test_pca)
    results["Linear Regression (PCA)"] = {
        "MAE": mean_absolute_error(y_test, lp),
        "RMSE": np.sqrt(mean_squared_error(y_test, lp)),
        "R2": r2_score(y_test, lp),
    }

    # 2) Random Forest (unscaled — tree models don't need scaling)
    rf = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=300)
    rf.fit(X_train_enc, y_train)
    rp = rf.predict(X_test_enc)
    results["Random Forest"] = {
        "MAE": mean_absolute_error(y_test, rp),
        "RMSE": np.sqrt(mean_squared_error(y_test, rp)),
        "R2": r2_score(y_test, rp),
    }

    # 3) XGBoost — the best model
    xgb = XGBRegressor(random_state=42).fit(X_train_scaled, y_train)
    xp = xgb.predict(X_test_scaled)
    results["XGBoost"] = {
        "MAE": mean_absolute_error(y_test, xp),
        "RMSE": np.sqrt(mean_squared_error(y_test, xp)),
        "R2": r2_score(y_test, xp),
    }

    cv_scores = cross_val_score(
        XGBRegressor(random_state=42), X_train_scaled, y_train, cv=5, scoring="r2"
    )

    return {
        "encoder": encoder,
        "scaler": scaler,
        "model": xgb,
        "feature_cols": X.columns.tolist(),
        "results": results,
        "cv_scores": cv_scores,
        "y_test": y_test,
        "xgb_pred": xp,
        "y_train": y_train,
        "train_pred": xgb.predict(X_train_scaled),
        "pca_components": pca.n_components_,
        "n_features": X_train_enc.shape[1],
        "importances": pd.Series(xgb.feature_importances_, index=X_train_enc.columns),
    }


# ----------------------------------------------------------------------------
# App body
# ----------------------------------------------------------------------------
raw = load_raw()
if raw is None:
    st.error("Could not find `laptopData.csv`. Upload it below to continue.")
    up = st.file_uploader("Upload laptopData.csv", type="csv")
    if up is None:
        st.stop()
    raw = pd.read_csv(up)

df, eda_df, options, before, after, inches_before_fix = clean_and_engineer(raw)
art = train_models(df)

st.sidebar.title("Laptop Price ML App")
page = st.sidebar.radio(
    "Navigate the workflow",
    [
        "1 — Data Overview",
        "2 — EDA",
        "3 — Cleaning & Preprocessing",
        "4 — Model Results",
        "5 — Predict a Price",
    ],
)
st.sidebar.markdown("---")
st.sidebar.caption(
    "Regression project: predicting laptop prices from hardware specs. "
    "Dataset: Uncleaned Laptop Price (Kaggle). Best model: XGBoost."
)

# ============================================================= 1. DATA OVERVIEW
if page.startswith("1"):
    st.markdown('<p class="main-title">Laptop Price Prediction</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="subtitle">A regression project: estimating a laptop\'s market price (INR) from its hardware specifications.</p>',
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f'<div class="metric-card"><h3>{raw.shape[0]:,}</h3><p>Raw rows</p></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="metric-card"><h3>{raw.shape[1]}</h3><p>Raw columns</p></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="metric-card"><h3>{df.shape[0]:,}</h3><p>Rows after cleaning</p></div>', unsafe_allow_html=True)
    with c4:
        st.markdown(f'<div class="metric-card"><h3>{art["results"]["XGBoost"]["R2"]*100:.1f}%</h3><p>Best model R²</p></div>', unsafe_allow_html=True)

    st.markdown("### The problem")
    st.write(
        "Laptop prices depend on a mix of brand, screen, CPU, GPU, RAM, and storage. "
        "The dataset arrives *uncleaned*: mixed units (`8GB`, `1.37kg`), `?` placeholders, "
        "fully-empty rows, and data-entry errors. The goal is to clean it, engineer useful "
        "features, and train a regression model that predicts the price."
    )

    st.markdown("### Sample of the raw dataset")
    st.dataframe(raw.sample(8, random_state=42), use_container_width=True)

    st.markdown("### Column dictionary")
    col_desc = pd.DataFrame(
        {
            "Column": ["Company", "TypeName", "Inches", "ScreenResolution", "Cpu", "Ram",
                       "Memory", "Gpu", "OpSys", "Weight", "Price"],
            "Description": [
                "Laptop manufacturer (Dell, HP, Apple, ...)",
                "Category: Notebook, Gaming, Ultrabook, 2 in 1 Convertible, ...",
                "Screen size in inches (contains '?' and entry errors like 35.6)",
                "Raw string mixing resolution, IPS panel, and touchscreen info",
                "Raw CPU string: brand + series + clock speed (GHz)",
                "RAM size as text, e.g. '8GB'",
                "Storage string, e.g. '128GB SSD + 1TB HDD'",
                "Raw GPU string: brand + series",
                "Operating system",
                "Weight as text, e.g. '1.37kg'",
                "Target — laptop price in Indian Rupees (INR)",
            ],
        }
    )
    st.dataframe(col_desc, use_container_width=True, hide_index=True)

# ======================================================================= 2. EDA
elif page.startswith("2"):
    st.header("Exploratory Data Analysis")

    # --- Target distribution ---
    st.subheader("Target: Price distribution")
    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(eda_df["Price"], bins=30, color="steelblue", edgecolor="white")
        ax.set_title("Price (original scale)", fontweight="bold")
        ax.set_xlabel("Price (INR)")
        st.pyplot(fig)
    with c2:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(np.log(eda_df["Price"]), bins=30, color="mediumseagreen", edgecolor="white")
        ax.set_title("log(Price)", fontweight="bold")
        ax.set_xlabel("log Price")
        st.pyplot(fig)
    st.info(
        "Price is strongly right-skewed, so the target was log-transformed before modeling. "
        "That gives the models a near-normal target and noticeably better fits."
    )

    # --- Numeric feature distributions ---
    st.subheader("Numeric feature distributions (standardized)")
    fig, ax = plt.subplots(figsize=(8, 4))
    for col in ["Inches", "Ram", "CpuGHz", "Weight", "MemoryGB"]:
        vals = pd.to_numeric(eda_df[col], errors="coerce").dropna()
        ax.hist((vals - vals.mean()) / vals.std(), bins=30, alpha=0.5, label=col)
    ax.set_title("Feature Distributions (Normalized)", fontweight="bold")
    ax.set_xlabel("Standardized value")
    ax.legend(fontsize=9)
    st.pyplot(fig)
    st.caption(
        "Plotted on a common standardized scale to compare shapes: RAM and MemoryGB are "
        "discrete and right-skewed, while Inches and Weight cluster around typical 15\" laptops."
    )

    # --- Price by RAM ---
    st.subheader("Median price by RAM")
    fig, ax = plt.subplots(figsize=(8, 4))
    ram_price = eda_df.groupby("Ram")["Price"].median().sort_index()
    ax.bar(ram_price.index.astype(str), ram_price.values, color="mediumpurple", edgecolor="white", width=0.6)
    ax.set_xlabel("RAM (GB)")
    ax.set_ylabel("Median Price (INR)")
    ax.set_title("Median Price by RAM", fontweight="bold")
    st.pyplot(fig)
    st.caption("A clear monotonic relationship: RAM is one of the strongest single price drivers.")

    # --- Correlation with price ---
    st.subheader("Correlation with Price")
    fig, ax = plt.subplots(figsize=(5, 8))
    num_df = df.select_dtypes(include=np.number)
    corr = num_df.corr()[["Price"]].sort_values("Price", ascending=False).head(15)
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, linewidths=0.5, ax=ax)
    ax.set_title("Top correlations with Price", fontweight="bold")
    st.pyplot(fig)
    st.caption(
        "RAM, screen resolution, CPU clock speed, and SSD presence show the strongest "
        "positive correlations with price."
    )

    # --- Price by company ---
    st.subheader("Median price by company")
    fig, ax = plt.subplots(figsize=(9, 4))
    comp_price = eda_df.groupby("Company")["Price"].median().sort_values(ascending=False)
    ax.bar(comp_price.index, comp_price.values, color="coral", edgecolor="white")
    ax.tick_params(axis="x", rotation=60)
    ax.set_ylabel("Median Price (INR)")
    st.pyplot(fig)
    st.caption("Premium brands (Razer, Apple, MSI) sit well above budget brands — brand carries real signal.")

# ====================================================== 3. CLEANING & PREPROCESSING
elif page.startswith("3"):
    st.header("Cleaning & Preprocessing Summary")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Before")
        st.markdown(
            f"""
            - Shape: **{before['shape'][0]:,} rows x {before['shape'][1]} columns**
            - Missing values: **{before['missing']:,}**
            - Duplicated rows: **{before['duplicates']}**
            - Mixed-unit text columns: `Ram` ('8GB'), `Weight` ('1.37kg'), `Memory` ('1TB HDD')
            - `?` placeholders in `Inches` and `Weight`
            - Data-entry errors: screen sizes like 35.6"
            """
        )
    with c2:
        st.markdown("#### After")
        st.markdown(
            f"""
            - Shape: **{after['shape'][0]:,} rows x {after['shape'][1]} columns**
            - Missing values: **{after['missing']}**
            - Duplicated rows: **{after['duplicates']}**
            - All features numeric and model-ready
            - 14 engineered features extracted from raw text
            """
        )

    st.markdown("---")
    st.subheader("Step-by-step decisions")
    st.markdown(
        """
        **1. Dropped useless rows/columns** — `Unnamed: 0` (just row numbers), fully-NaN rows
        (nothing to impute from), and exact duplicates.

        **2. `Inches`** — replaced `?` with NaN, imputed with the **median** (robust to outliers),
        then fixed data-entry errors: values like 25.6" and 35.6" belonged to laptops whose
        weights matched normal 15.6" machines, so they were corrected (25.6 → 15.6, 27.3 → 17.3, ...).

        **3. `ScreenResolution`** — split into 4 features: `ScreenWidth`, `ScreenHeight`,
        `IPS` (0/1), `TouchScreen` (0/1).

        **4. `Cpu`** — extracted `CpuBrand`, `CpuGHz` (numeric), and grouped the series into a
        clean `CpuFamily` (Core i5, Ryzen, Celeron, ...).

        **5. `Weight`** — replaced `?`, removed the 'kg' unit, imputed with the **median**.

        **6. `Memory`** — unified TB → GB, imputed `?` with the **mode** (column was still text),
        created binary flags `SSD` / `HDD` / `Flash_Storage` / `Hybrid`, and summed multi-drive
        strings ('128GB SSD + 1TB HDD') into a total `MemoryGB`.

        **7. `Gpu`** — extracted `Gpu_Brand` and a grouped `GpuSeries` (RTX, GTX, Radeon, HD, ...).

        **8. Encoding** — one-hot encoding for most categoricals; **BinaryEncoder** for `Company`
        (19 brands → a few binary columns instead of 19 dummy columns). The encoder was fit on
        the *training set only* to avoid data leakage.

        **9. Target transform** — `log(Price)` to fix strong right-skewness.

        **10. Scaling** — `StandardScaler` fit on the training set only (needed for the linear
        model; tree models like Random Forest don't require it).
        """
    )

    st.markdown("---")
    st.subheader("Before / after example: fixing `Inches`")
    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.boxplot(inches_before_fix, patch_artist=True,
                   boxprops=dict(facecolor="indianred", alpha=0.6),
                   medianprops=dict(color="darkred", linewidth=2))
        ax.set_title("Before fix — impossible sizes up to 35.6\"", fontweight="bold", fontsize=10)
        ax.set_ylabel("Inches")
        st.pyplot(fig)
    with c2:
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.boxplot(df["Inches"], patch_artist=True,
                   boxprops=dict(facecolor="steelblue", alpha=0.6),
                   medianprops=dict(color="red", linewidth=2))
        ax.set_title("After fix — realistic 10\" to 18\" range", fontweight="bold", fontsize=10)
        ax.set_ylabel("Inches")
        st.pyplot(fig)

    st.markdown("---")
    st.subheader("Data preview: raw vs processed")
    tab1, tab2 = st.tabs(["Raw data", "Processed (model-ready)"])
    with tab1:
        st.dataframe(raw.head(6), use_container_width=True)
    with tab2:
        st.dataframe(df.head(6), use_container_width=True)

# ================================================================ 4. MODEL RESULTS
elif page.startswith("4"):
    st.header("Model Results")
    st.write(
        "Three models were trained on an 80/20 split with `log(Price)` as the target. "
        "PCA (95% variance) was tested with Linear Regression but *reduced* performance, "
        "so the tree-based models use the full scaled feature set."
    )

    res = pd.DataFrame(art["results"]).T.round(3).sort_values("R2", ascending=False)
    st.dataframe(
        res.style.highlight_max(subset=["R2"], color="#14532d")
                 .highlight_min(subset=["MAE", "RMSE"], color="#14532d"),
        use_container_width=True,
    )
    st.caption("Metrics are computed on the log-price scale, on the held-out test set.")

    xgb_r2 = art["results"]["XGBoost"]["R2"]
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f'<div class="metric-card"><h3>{xgb_r2*100:.1f}%</h3><p>XGBoost test R²</p></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="metric-card"><h3>{art["results"]["XGBoost"]["MAE"]:.3f}</h3><p>MAE (log scale)</p></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="metric-card"><h3>{art["results"]["XGBoost"]["RMSE"]:.3f}</h3><p>RMSE (log scale)</p></div>', unsafe_allow_html=True)
    with c4:
        st.markdown(f'<div class="metric-card"><h3>{art["cv_scores"].mean():.3f}</h3><p>5-fold CV mean R²</p></div>', unsafe_allow_html=True)

    st.markdown("### Actual vs Predicted — XGBoost")
    c1, c2 = st.columns([3, 2])
    with c1:
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(art["y_test"], art["xgb_pred"], alpha=0.5, color="steelblue")
        lims = [art["y_test"].min(), art["y_test"].max()]
        ax.plot(lims, lims, "r--", linewidth=2)
        ax.set_xlabel("Actual Price (log)")
        ax.set_ylabel("Predicted Price (log)")
        ax.set_title("Actual vs Predicted — XGBoost", fontweight="bold")
        st.pyplot(fig)
    with c2:
        st.markdown("#### Reading the plot")
        st.write(
            "Points hug the red diagonal across the whole price range, meaning the model is "
            "accurate for both budget and premium laptops, with slightly more spread at the "
            "extremes where the data is sparser."
        )
        st.markdown("#### Cross-validation")
        st.write(
            f"5-fold CV R² scores: {np.round(art['cv_scores'], 3).tolist()} — "
            f"stable around **{art['cv_scores'].mean():.3f}**, so the test score is not a lucky split."
        )

    st.markdown("### Top 15 feature importances (XGBoost)")
    fig, ax = plt.subplots(figsize=(8, 5))
    top = art["importances"].sort_values(ascending=True).tail(15)
    ax.barh(top.index, top.values, color="mediumpurple", edgecolor="white")
    ax.set_xlabel("Importance")
    st.pyplot(fig)
    st.caption("RAM, screen resolution, CPU speed, and workstation/gaming type dominate the price signal.")

# ============================================================ 5. LIVE PREDICTION
else:
    st.header("Predict a Laptop Price")
    st.write("Configure a laptop below and the trained XGBoost model returns a live price estimate.")

    with st.container():
        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("#### General")
            company = st.selectbox("Brand", options["Company"], index=options["Company"].index("Dell") if "Dell" in options["Company"] else 0)
            typename = st.selectbox("Type", options["TypeName"])
            opsys = st.selectbox("Operating system", options["OpSys"])
            weight = st.slider("Weight (kg)", 0.7, 4.7, 1.8, 0.1)
        with c2:
            st.markdown("#### Display")
            inches = st.slider("Screen size (inches)", 10.1, 18.4, 15.6, 0.1)
            resolution = st.selectbox("Resolution", options["Resolutions"],
                                      index=options["Resolutions"].index("1920 x 1080") if "1920 x 1080" in options["Resolutions"] else 0)
            ips = st.checkbox("IPS panel")
            touch = st.checkbox("Touchscreen")
        with c3:
            st.markdown("#### Performance")
            ram = st.select_slider("RAM (GB)", options=options["Ram"], value=8)
            cpu_brand = st.selectbox("CPU brand", options["CpuBrand"])
            cpu_fam = st.selectbox("CPU family", options["CpuFamily"],
                                   index=options["CpuFamily"].index("Core i5") if "Core i5" in options["CpuFamily"] else 0)
            cpu_ghz = st.slider("CPU clock speed (GHz)", 0.9, 3.6, 2.5, 0.1)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Storage")
        storage_types = st.multiselect("Storage type(s)", ["SSD", "HDD", "Flash Storage", "Hybrid"], default=["SSD"])
        memory_gb = st.select_slider("Total storage (GB)", options=[32, 64, 128, 256, 512, 1024, 1536, 2048, 2560], value=512)
    with c2:
        st.markdown("#### Graphics")
        gpu_brand = st.selectbox("GPU brand", options["Gpu_Brand"])
        gpu_ser = st.selectbox("GPU series", options["GpuSeries"])
        currency = st.radio("Show price in", ["INR", "SAR", "USD"], horizontal=True)

    if st.button("Predict price", type="primary", use_container_width=True):
        width, height = (int(v) for v in resolution.split(" x "))

        # build one row with every training feature = 0, then fill it in
        row = {col: 0 for col in art["feature_cols"]}
        row.update({
            "Company": company,          # BinaryEncoder handles the raw string
            "Inches": inches,
            "Ram": ram,
            "Weight": weight,
            "ScreenHeight": height,
            "ScreenWidth": width,
            "IPS": int(ips),
            "TouchScreen": int(touch),
            "CpuGHz": cpu_ghz,
            "SSD": int("SSD" in storage_types),
            "HDD": int("HDD" in storage_types),
            "Flash_Storage": int("Flash Storage" in storage_types),
            "Hybrid": int("Hybrid" in storage_types),
            "MemoryGB": memory_gb,
        })
        # one-hot columns — only set the ones that exist in training
        for col in (f"TypeName_{typename}", f"OpSys_{opsys}", f"CpuBrand_{cpu_brand}",
                    f"Gpu_Brand_{gpu_brand}", f"CpuFamily_{cpu_fam}", f"GpuSeries_{gpu_ser}"):
            if col in row:
                row[col] = 1

        input_df = pd.DataFrame([row])[art["feature_cols"]]
        input_enc = art["encoder"].transform(input_df)
        input_scaled = art["scaler"].transform(input_enc)
        log_price = art["model"].predict(input_scaled)[0]
        price_inr = float(np.exp(log_price))   # invert the log transform

        rates = {"INR": (1.0, "₹"), "SAR": (0.045, "SAR "), "USD": (0.012, "$")}
        rate, symbol = rates[currency]
        shown = price_inr * rate

        st.markdown(
            f"""
            <div class="price-box">
              <p style="margin:0;color:#8a8f98;">Estimated price</p>
              <h2>{symbol}{shown:,.0f}</h2>
              <p style="margin:0;color:#8a8f98;">{company} · {typename} · {ram}GB RAM · {memory_gb}GB · {resolution}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.caption(
            "The model predicts log(Price) in INR and the app converts back with exp(). "
            "SAR/USD figures use approximate fixed exchange rates for display only."
        )
