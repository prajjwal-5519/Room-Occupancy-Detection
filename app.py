import io
import os
import urllib.request
import zipfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from nb_models import KernelNaiveBayes

st.set_page_config(page_title="Room Occupancy - Naive Bayes", page_icon="🏢", layout="wide")

FEATURES = ["Temperature", "Humidity", "Light", "CO2", "HumidityRatio"]
CSV_PATH = "occupancy_dataset.csv"
URL = "https://archive.ics.uci.edu/static/public/357/occupancy+detection.zip"
RANDOM_STATE = 42


# ----------------------------------------------------------------------------- data
@st.cache_data(show_spinner="Loading dataset...")
def load_data() -> pd.DataFrame:
    """Use occupancy_dataset.csv from the repo; otherwise download from UCI and cache it."""
    if not os.path.exists(CSV_PATH):
        raw = urllib.request.urlopen(URL, timeout=60).read()
        frames = []
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            for name in z.namelist():
                if name.endswith(".txt"):
                    part = pd.read_csv(z.open(name))
                    frames.append(part.loc[:, ~part.columns.str.contains("^Unnamed")])
        pd.concat(frames, ignore_index=True).to_csv(CSV_PATH, index=False)
    return pd.read_csv(CSV_PATH)


def humidity_ratio(temp_c: float, rh_percent: float, pressure_hpa: float = 1013.25) -> float:
    """Approximate humidity ratio (kg water vapour / kg dry air) from temperature and RH."""
    p_sat = 6.112 * np.exp(17.62 * temp_c / (243.12 + temp_c))
    p_w = rh_percent / 100.0 * p_sat
    return float(0.622 * p_w / (pressure_hpa - p_w))


# --------------------------------------------------------------------------- models
@st.cache_resource(show_spinner="Training Naive Bayes models (80% train)...")
def train_models(bandwidth: float):
    df = load_data()
    X, y = df[FEATURES], df["Occupancy"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )
    models = {
        "Gaussian NB": make_pipeline(StandardScaler(), GaussianNB()),
        "Kernel NB (KDE)": make_pipeline(StandardScaler(), KernelNaiveBayes(bandwidth=bandwidth)),
    }
    preds = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds[name] = model.predict(X_test)
    return models, y_test.to_numpy(), preds, len(X_train), len(X_test)


def compute_metrics(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return cm, {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Type I Error (FPR)": fp / (fp + tn),
        "Type II Error (FNR)": fn / (fn + tp),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1 Score": f1_score(y_true, y_pred),
    }


def heatmaps(cm, title):
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
    labels_x = ["Pred: Empty", "Pred: Occupied"]
    labels_y = ["Actual: Empty", "Actual: Occupied"]
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax[0], xticklabels=labels_x, yticklabels=labels_y)
    ax[0].set_title(f"{title}\nCounts")
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    sns.heatmap(cm_norm, annot=True, fmt=".3f", cmap="YlGnBu", ax=ax[1], xticklabels=labels_x, yticklabels=labels_y)
    ax[1].set_title(f"{title}\nRow-normalised rates")
    fig.tight_layout()
    return fig


# ------------------------------------------------------------------------------- UI
st.title("🏢 Room Occupancy Detection from Sensor Data")
st.caption(
    "Gaussian and Kernel Naive Bayes classifiers on the UCI Occupancy Detection dataset "
    "(Temperature, Humidity, Light, CO2, Humidity Ratio) - 80/20 train-test split."
)

try:
    df = load_data()
except Exception as exc:  # network or file problem
    st.error(
        "Could not load the dataset. Add `occupancy_dataset.csv` to the app repository "
        f"or check network access. Details: {exc}"
    )
    st.stop()

with st.sidebar:
    st.header("Settings")
    bandwidth = st.slider("Kernel NB bandwidth (KDE)", 0.05, 0.50, 0.10, 0.05)
    st.markdown(
        "**Positive class:** Occupied (1)\n\n"
        "**Type I error:** empty room predicted occupied (FP)\n\n"
        "**Type II error:** occupied room predicted empty (FN)"
    )

models, y_test, preds, n_train, n_test = train_models(bandwidth)

tab_live, tab_eval, tab_data = st.tabs(["🔴 Live prediction", "📊 Evaluation", "🗂 Dataset"])

# ---- live prediction
with tab_live:
    st.subheader("Enter sensor readings")
    c1, c2 = st.columns(2)
    with c1:
        temp = st.slider("Temperature (°C)", 15.0, 30.0, 21.0, 0.1)
        hum = st.slider("Relative humidity (%)", 10.0, 45.0, 27.0, 0.1)
        hr = humidity_ratio(temp, hum)
        st.metric("Humidity ratio (derived)", f"{hr:.6f}")
    with c2:
        light = st.slider("Light (lux)", 0, 1700, 100, 5)
        co2 = st.slider("CO2 (ppm)", 350, 2100, 500, 10)

    sample = pd.DataFrame([[temp, hum, light, co2, hr]], columns=FEATURES)
    cols = st.columns(len(models))
    for col, (name, model) in zip(cols, models.items()):
        p_occ = float(model.predict_proba(sample)[0, 1])
        label = "OCCUPIED" if p_occ >= 0.5 else "NOT OCCUPIED"
        with col:
            st.markdown(f"##### {name}")
            (st.success if p_occ >= 0.5 else st.info)(f"**{label}**")
            st.progress(min(max(p_occ, 0.0), 1.0), text=f"P(occupied) = {p_occ:.3f}")

# ---- evaluation
with tab_eval:
    st.write(f"Training samples: **{n_train}** (80%)  |  Testing samples: **{n_test}** (20%)")
    results, cms = {}, {}
    for name in models:
        cms[name], results[name] = compute_metrics(y_test, preds[name])
    st.dataframe(pd.DataFrame(results).T.style.format("{:.4f}"))

    for name in models:
        tn, fp, fn, tp = cms[name].ravel()
        st.markdown(f"#### {name}")
        st.caption(f"TN = {tn}, FP = {fp} (Type I), FN = {fn} (Type II), TP = {tp}")
        fig = heatmaps(cms[name], name)
        st.pyplot(fig)
        plt.close(fig)

# ---- dataset
with tab_data:
    st.write(f"**{len(df)} rows**, features: {', '.join(FEATURES)}; label: `Occupancy` (0/1)")
    st.dataframe(df.head(100))
    st.bar_chart(df["Occupancy"].value_counts().rename({0: "Not occupied", 1: "Occupied"}))
    st.caption(
        "Source: L. Candanedo, UCI Machine Learning Repository, doi:10.24432/C5X01N (CC BY 4.0)."
    )
