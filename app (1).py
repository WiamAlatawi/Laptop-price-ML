"""
Laptop Price Prediction — Interactive Streamlit App
====================================================
Deploys the full ML workflow from ML_laptop_price.ipynb as an interactive web app:
1. Data Overview  2. EDA  3. Cleaning & Preprocessing  4. Model Results  5. Live Prediction

All charts are interactive (Plotly): hover, zoom, pan, and download.
The prediction page lets the visitor choose between the two best models
(XGBoost and Random Forest) before predicting.

Run locally:   streamlit run app.py
Dataset:       laptopData.csv (place next to app.py, or it downloads via kagglehub)
"""

import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

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

PLOTLY_TEMPLATE = "plotly_white"
ACCENT = "#6366f1"
GREEN = "#10b981"
CORAL = "#f97362"
PURPLE = "#a78bfa"


def style_fig(fig, height=420):
    """Shared interactive-chart styling."""
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        height=height,
        margin=dict(l=10, r=10, t=50, b=10),
        title_font=dict(size=16),
        hoverlabel=dict(font_size=13),
    )
    return fig


# ----------------------------------------------------------------------------
# 0. Data loading
# ----------------------------------------------------------------------------
@st.cache_data(show_spinner="Loading dataset...")
def load_raw():
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
    eda_df = df.copy()  # readable copy (pre one-hot) for the interactive EDA

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
    }
    return df, eda_df, options, before, after, inches_before_fix


@st.cache_resource(show_spinner="Training models (first run only)...")
def train_models(df: pd.DataFrame):
    """Split, encode Company (BinaryEncoder), scale, train the 3 notebook models.
    Both XGBoost and Random Forest are kept so the visitor can choose either one."""
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

    results, preds = {}, {}

    # 1) Linear Regression on PCA components (95% variance) — baseline
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
    preds["Linear Regression (PCA)"] = lp

    # 2) Random Forest (unscaled — tree models don't need scaling)
    rf = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=300)
    rf.fit(X_train_enc, y_train)
    rp = rf.predict(X_test_enc)
    results["Random Forest"] = {
        "MAE": mean_absolute_error(y_test, rp),
        "RMSE": np.sqrt(mean_squared_error(y_test, rp)),
        "R2": r2_score(y_test, rp),
    }
    preds["Random Forest"] = rp

    # 3) XGBoost — the best model in the notebook
    xgb = XGBRegressor(random_state=42).fit(X_train_scaled, y_train)
    xp = xgb.predict(X_test_scaled)
    results["XGBoost"] = {
        "MAE": mean_absolute_error(y_test, xp),
        "RMSE": np.sqrt(mean_squared_error(y_test, xp)),
        "R2": r2_score(y_test, xp),
    }
    preds["XGBoost"] = xp

    cv_scores = cross_val_score(
        XGBRegressor(random_state=42), X_train_scaled, y_train, cv=5, scoring="r2"
    )

    return {
        "encoder": encoder,
        "scaler": scaler,
        "models": {"XGBoost": xgb, "Random Forest": rf},   # the two choices
        "feature_cols": X.columns.tolist(),
        "results": results,
        "preds": preds,
        "cv_scores": cv_scores,
        "y_test": y_test,
        "importances": {
            "XGBoost": pd.Series(xgb.feature_importances_, index=X_train_enc.columns),
            "Random Forest": pd.Series(rf.feature_importances_, index=X_train_enc.columns),
        },
        "n_features": X_train_enc.shape[1],
        "pca_components": pca.n_components_,
    }


def predict_price(art: dict, model_name: str, row: dict) -> float:
    """Encode + (scale if needed) + predict, then invert the log transform."""
    input_df = pd.DataFrame([row])[art["feature_cols"]]
    input_enc = art["encoder"].transform(input_df)
    if model_name == "XGBoost":                 # trained on scaled features
        input_enc = art["scaler"].transform(input_enc)
    log_price = art["models"][model_name].predict(input_enc)[0]
    return float(np.exp(log_price))


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
    "Dataset: Uncleaned Laptop Price (Kaggle). All charts are interactive — "
    "hover for details, drag to zoom, double-click to reset."
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

    st.markdown("### Explore a sample of the raw dataset")
    n_sample = st.slider("Sample size", 5, 30, 10)
    st.dataframe(raw.sample(n_sample, random_state=42), use_container_width=True)

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
    st.caption("Every chart is interactive: hover for exact values, drag to zoom, double-click to reset.")

    # --- Target distribution (log toggle) ---
    st.subheader("Target: Price distribution")
    log_view = st.toggle("Show log(Price)", value=False)
    if log_view:
        fig = px.histogram(np.log(eda_df["Price"]), nbins=30,
                           color_discrete_sequence=[GREEN],
                           labels={"value": "log Price"}, title="log(Price) — near-normal after transform")
    else:
        fig = px.histogram(eda_df, x="Price", nbins=30,
                           color_discrete_sequence=["#4682b4"],
                           title="Price (INR) — strongly right-skewed")
    fig.update_layout(showlegend=False)
    st.plotly_chart(style_fig(fig), use_container_width=True)
    st.info(
        "Price is strongly right-skewed, so the target was log-transformed before modeling. "
        "Flip the toggle to see how the log transform makes it near-normal."
    )

    # --- Pick-a-feature distribution ---
    st.subheader("Numeric feature distributions")
    num_features = ["Inches", "Ram", "CpuGHz", "Weight", "MemoryGB", "ScreenWidth", "ScreenHeight"]
    chosen = st.multiselect("Choose features to compare (standardized)", num_features,
                            default=["Inches", "Ram", "CpuGHz", "Weight", "MemoryGB"])
    fig = go.Figure()
    for col in chosen:
        vals = pd.to_numeric(eda_df[col], errors="coerce").dropna()
        fig.add_trace(go.Histogram(x=(vals - vals.mean()) / vals.std(), name=col, opacity=0.55, nbinsx=30))
    fig.update_layout(barmode="overlay", title="Feature Distributions (Normalized)",
                      xaxis_title="Standardized value", yaxis_title="Count")
    st.plotly_chart(style_fig(fig), use_container_width=True)
    st.caption(
        "Plotted on a common standardized scale to compare shapes: RAM and MemoryGB are "
        "discrete and right-skewed, while Inches and Weight cluster around typical 15\" laptops."
    )

    # --- Price vs any feature ---
    st.subheader("How does price relate to a spec?")
    c1, c2 = st.columns([1, 3])
    with c1:
        x_feat = st.selectbox("Feature", ["Ram", "Company", "TypeName", "CpuFamily", "GpuSeries",
                                          "Inches", "CpuGHz", "Weight", "MemoryGB", "ScreenWidth"])
        agg = st.radio("View", ["Median bar", "Box plot", "Scatter"],
                       index=0 if eda_df[x_feat].dtype == object or x_feat == "Ram" else 2)
    with c2:
        if agg == "Median bar":
            data = eda_df.groupby(x_feat)["Price"].median().sort_values(ascending=False).reset_index()
            fig = px.bar(data, x=x_feat, y="Price", color="Price",
                         color_continuous_scale="Purples", title=f"Median Price by {x_feat}")
        elif agg == "Box plot":
            fig = px.box(eda_df, x=x_feat, y="Price", color_discrete_sequence=[PURPLE],
                         title=f"Price distribution by {x_feat}")
        else:
            fig = px.scatter(eda_df, x=x_feat, y="Price", color="TypeName",
                             hover_data=["Company", "Ram", "CpuFamily"],
                             opacity=0.6, title=f"Price vs {x_feat}")
        st.plotly_chart(style_fig(fig, 460), use_container_width=True)
    st.caption(
        "Try RAM (clear monotonic driver), Company (Razer/Apple/MSI at the top), or a scatter "
        "of CpuGHz colored by laptop type."
    )

    # --- Correlation with price ---
    st.subheader("Correlation with Price")
    top_n = st.slider("Show top N correlated features", 5, 25, 15)
    num_df = df.select_dtypes(include=np.number)
    corr = num_df.corr()["Price"].drop("Price").sort_values(ascending=False)
    corr_view = pd.concat([corr.head(top_n // 2 + top_n % 2), corr.tail(top_n // 2)])
    fig = px.bar(corr_view[::-1], orientation="h",
                 color=corr_view[::-1], color_continuous_scale="RdBu_r", range_color=[-1, 1],
                 labels={"value": "Correlation with Price", "index": ""},
                 title="Strongest positive & negative correlations with Price")
    fig.update_layout(showlegend=False, coloraxis_showscale=False)
    st.plotly_chart(style_fig(fig, 520), use_container_width=True)
    st.caption(
        "RAM, screen resolution, CPU clock speed, and SSD presence show the strongest positive "
        "correlations; Notebook type and HDD-only storage pull prices down."
    )

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
    steps = {
        "1 · Drop useless rows/columns": "`Unnamed: 0` (just row numbers), fully-NaN rows (nothing to impute from), and exact duplicates.",
        "2 · Inches": "Replaced `?` with NaN and imputed with the **median** (robust to outliers). Then fixed data-entry errors: sizes like 25.6\" and 35.6\" belonged to laptops whose weights matched normal 15.6\" machines, so they were corrected (25.6 → 15.6, 27.3 → 17.3, ...).",
        "3 · ScreenResolution": "Split into 4 features: `ScreenWidth`, `ScreenHeight`, `IPS` (0/1), `TouchScreen` (0/1).",
        "4 · Cpu": "Extracted `CpuBrand`, numeric `CpuGHz`, and grouped the series into a clean `CpuFamily` (Core i5, Ryzen, Celeron, ...).",
        "5 · Weight": "Replaced `?`, removed the 'kg' unit, imputed with the **median**.",
        "6 · Memory": "Unified TB → GB, imputed `?` with the **mode** (column was still text), created binary flags `SSD` / `HDD` / `Flash_Storage` / `Hybrid`, and summed multi-drive strings ('128GB SSD + 1TB HDD') into a total `MemoryGB`.",
        "7 · Gpu": "Extracted `Gpu_Brand` and a grouped `GpuSeries` (RTX, GTX, Radeon, HD, ...).",
        "8 · Encoding": "One-hot encoding for most categoricals; **BinaryEncoder** for `Company` (19 brands → a few binary columns instead of 19 dummies). Fit on the *training set only* — no data leakage.",
        "9 · Target transform": "`log(Price)` to fix the strong right-skewness.",
        "10 · Scaling": "`StandardScaler` fit on the training set only (needed for the linear model; tree models don't require it).",
    }
    for title, body in steps.items():
        with st.expander(title):
            st.markdown(body)

    st.markdown("---")
    st.subheader("Before / after example: fixing `Inches`")
    fig = go.Figure()
    fig.add_trace(go.Box(y=inches_before_fix, name="Before fix", marker_color=CORAL,
                         boxpoints="outliers"))
    fig.add_trace(go.Box(y=df["Inches"], name="After fix", marker_color="#4682b4",
                         boxpoints="outliers"))
    fig.update_layout(title="Inches — impossible 24\"–35.6\" values corrected to the real 10\"–18\" range",
                      yaxis_title="Inches")
    st.plotly_chart(style_fig(fig, 440), use_container_width=True)
    st.caption("Hover the outlier points on the left box to see the exact impossible values that were fixed.")

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
        "so the tree-based models use the full feature set."
    )

    res = pd.DataFrame(art["results"]).T.round(3).sort_values("R2", ascending=False)

    # interactive metric comparison
    metric = st.radio("Compare models by", ["R2", "MAE", "RMSE"], horizontal=True)
    order = res[metric].sort_values(ascending=(metric != "R2"))
    fig = px.bar(order, orientation="h", text_auto=".3f",
                 color=order.index,
                 color_discrete_sequence=[GREEN, ACCENT, CORAL],
                 labels={"value": f"{metric} (log-price scale)", "index": ""},
                 title=f"Model comparison — {metric} on the held-out test set")
    fig.update_layout(showlegend=False)
    st.plotly_chart(style_fig(fig, 320), use_container_width=True)
    st.dataframe(res, use_container_width=True)
    st.caption("Metrics are computed on the log-price scale. Higher R² is better; lower MAE/RMSE is better.")

    xgb_r2 = art["results"]["XGBoost"]["R2"]
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f'<div class="metric-card"><h3>{xgb_r2*100:.1f}%</h3><p>XGBoost test R²</p></div>', unsafe_allow_html=True)
    with c2:
        st.markdown(f'<div class="metric-card"><h3>{art["results"]["Random Forest"]["R2"]*100:.1f}%</h3><p>Random Forest test R²</p></div>', unsafe_allow_html=True)
    with c3:
        st.markdown(f'<div class="metric-card"><h3>{art["cv_scores"].mean():.3f}</h3><p>XGBoost 5-fold CV R²</p></div>', unsafe_allow_html=True)
    with c4:
        st.markdown(f'<div class="metric-card"><h3>{art["n_features"]}</h3><p>Features after encoding</p></div>', unsafe_allow_html=True)

    st.markdown("### Actual vs Predicted")
    model_choice = st.selectbox("Model", ["XGBoost", "Random Forest", "Linear Regression (PCA)"])
    y_test = art["y_test"]
    pred = art["preds"][model_choice]
    resid = np.abs(y_test.values - pred)
    fig = px.scatter(
        x=y_test, y=pred, color=resid, color_continuous_scale="Tealrose",
        labels={"x": "Actual Price (log)", "y": "Predicted Price (log)", "color": "|error|"},
        title=f"Actual vs Predicted — {model_choice}", opacity=0.7,
    )
    lims = [float(y_test.min()), float(y_test.max())]
    fig.add_trace(go.Scatter(x=lims, y=lims, mode="lines",
                             line=dict(color="red", dash="dash", width=2),
                             name="Perfect prediction"))
    st.plotly_chart(style_fig(fig, 500), use_container_width=True)
    st.write(
        "Points hug the red diagonal across the whole price range for the tree models, meaning "
        "they're accurate for both budget and premium laptops. Hover any point to see its exact "
        "actual vs predicted log-price; the color shows the absolute error."
    )
    st.write(
        f"5-fold cross-validation for XGBoost: scores {np.round(art['cv_scores'], 3).tolist()}, "
        f"stable around **{art['cv_scores'].mean():.3f}** — so the test score is not a lucky split."
    )

    st.markdown("### Feature importances")
    imp_model = st.radio("Model", ["XGBoost", "Random Forest"], horizontal=True, key="imp")
    top_k = st.slider("Top features", 5, 25, 15)
    top = art["importances"][imp_model].sort_values(ascending=True).tail(top_k)
    fig = px.bar(top, orientation="h", color=top.values, color_continuous_scale="Purples",
                 labels={"value": "Importance", "index": ""},
                 title=f"Top {top_k} feature importances — {imp_model}")
    fig.update_layout(showlegend=False, coloraxis_showscale=False)
    st.plotly_chart(style_fig(fig, 520), use_container_width=True)
    st.caption("RAM, screen resolution, CPU speed, and workstation/gaming type dominate the price signal.")

# ============================================================ 5. LIVE PREDICTION
else:
    st.header("Predict a Laptop Price")
    st.write(
        "Configure a laptop, **choose one of the two best models**, and get a live price estimate."
    )

    # ---- model choice: one of the two tree models ----
    st.markdown("#### Step 1 — Choose your model")
    m1, m2 = st.columns(2)
    with m1:
        st.markdown(
            f"""<div class="metric-card"><h3>XGBoost</h3>
            <p>Test R² {art['results']['XGBoost']['R2']*100:.1f}% · the notebook's best model</p></div>""",
            unsafe_allow_html=True,
        )
    with m2:
        st.markdown(
            f"""<div class="metric-card"><h3>Random Forest</h3>
            <p>Test R² {art['results']['Random Forest']['R2']*100:.1f}% · strong ensemble baseline</p></div>""",
            unsafe_allow_html=True,
        )
    model_name = st.radio("Model to use for the prediction",
                          ["XGBoost", "Random Forest"], horizontal=True)

    st.markdown("#### Step 2 — Configure the laptop")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("##### General")
        company = st.selectbox("Brand", options["Company"],
                               index=options["Company"].index("Dell") if "Dell" in options["Company"] else 0)
        typename = st.selectbox("Type", options["TypeName"])
        opsys = st.selectbox("Operating system", options["OpSys"])
        weight = st.slider("Weight (kg)", 0.7, 4.7, 1.8, 0.1)
    with c2:
        st.markdown("##### Display")
        inches = st.slider("Screen size (inches)", 10.1, 18.4, 15.6, 0.1)
        resolution = st.selectbox("Resolution", options["Resolutions"],
                                  index=options["Resolutions"].index("1920 x 1080") if "1920 x 1080" in options["Resolutions"] else 0)
        ips = st.checkbox("IPS panel")
        touch = st.checkbox("Touchscreen")
    with c3:
        st.markdown("##### Performance")
        ram = st.select_slider("RAM (GB)", options=options["Ram"],
                               value=8 if 8 in options["Ram"] else options["Ram"][0])
        cpu_brand = st.selectbox("CPU brand", options["CpuBrand"])
        cpu_fam = st.selectbox("CPU family", options["CpuFamily"],
                               index=options["CpuFamily"].index("Core i5") if "Core i5" in options["CpuFamily"] else 0)
        cpu_ghz = st.slider("CPU clock speed (GHz)", 0.9, 3.6, 2.5, 0.1)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("##### Storage")
        storage_types = st.multiselect("Storage type(s)", ["SSD", "HDD", "Flash Storage", "Hybrid"], default=["SSD"])
        memory_gb = st.select_slider("Total storage (GB)",
                                     options=[32, 64, 128, 256, 512, 1024, 1536, 2048, 2560], value=512)
    with c2:
        st.markdown("##### Graphics")
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

        price_inr = predict_price(art, model_name, row)
        other = "Random Forest" if model_name == "XGBoost" else "XGBoost"
        price_other = predict_price(art, other, row)

        rates = {"INR": (1.0, "₹"), "SAR": (0.045, "SAR "), "USD": (0.012, "$")}
        rate, symbol = rates[currency]

        st.markdown(
            f"""
            <div class="price-box">
              <p style="margin:0;color:#8a8f98;">Estimated price · {model_name}</p>
              <h2>{symbol}{price_inr * rate:,.0f}</h2>
              <p style="margin:0;color:#8a8f98;">{company} · {typename} · {ram}GB RAM · {memory_gb}GB · {resolution}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # small interactive comparison of the two models on this exact laptop
        fig = px.bar(
            x=[model_name, other], y=[price_inr * rate, price_other * rate],
            color=[model_name, other], text_auto=",.0f",
            color_discrete_sequence=[GREEN, ACCENT],
            labels={"x": "", "y": f"Predicted price ({currency})"},
            title="How the two models price this exact configuration",
        )
        fig.update_layout(showlegend=False)
        st.plotly_chart(style_fig(fig, 340), use_container_width=True)

        st.caption(
            "The model predicts log(Price) in INR and the app converts back with exp(). "
            "SAR/USD figures use approximate fixed exchange rates for display only."
        )
