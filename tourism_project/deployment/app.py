"""
Streamlit app for Wellness Package Purchase prediction.

Expect environment variables:
- HF_TOKEN (recommended)
- HF_MODEL_REPO (e.g. "sathishaiuse/wellness-classifier-model")
- HF_MODEL_FILENAME (e.g. "best_tourism_model.joblib")

The app will:
- download model from HF model hub if not present locally
- accept a single-record input via form or CSV upload
- save incoming inputs to inputs/inputs.csv (append)
- run inference and display predictions (+ probability if available)
"""

import os
import shutil
from typing import Dict, Any

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from huggingface_hub import hf_hub_download

# ---------- CONFIG ----------
HF_TOKEN = os.environ.get("HF_TOKEN", None)
HF_MODEL_REPO = os.environ.get("HF_MODEL_REPO", "sathishaiuse/wellness-classifier-model")
HF_MODEL_FILENAME = os.environ.get("HF_MODEL_FILENAME", "best_tourism_model.joblib")
LOCAL_MODEL_DIR = "models"
LOCAL_MODEL_PATH = f"{LOCAL_MODEL_DIR}/{HF_MODEL_FILENAME}"

# Features — must match training pipeline
NUMERIC_FEATURES = [
    "Age", "CityTier", "NumberOfPersonVisiting", "PreferredPropertyStar",
    "NumberOfTrips", "Passport", "OwnCar", "NumberOfChildrenVisiting",
    "MonthlyIncome", "PitchSatisfactionScore", "NumberOfFollowups", "DurationOfPitch"
]
CATEGORICAL_FEATURES = [
    "TypeofContact", "Occupation", "Gender", "MaritalStatus", "Designation", "ProductPitched"
]
ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

os.makedirs(LOCAL_MODEL_DIR, exist_ok=True)
os.makedirs("inputs", exist_ok=True)

# ---------- MODEL LOADER ----------
@st.cache_resource
def download_and_load_model():
    # download model if not exists
    if not os.path.exists(LOCAL_MODEL_PATH):
        try:
            st.info(f"Downloading model {HF_MODEL_FILENAME} from {HF_MODEL_REPO} ...")
            cache_path = hf_hub_download(repo_id=HF_MODEL_REPO, filename=HF_MODEL_FILENAME,
                                        repo_type="model", use_auth_token=HF_TOKEN)
            shutil.copy(cache_path, LOCAL_MODEL_PATH)
            st.success("Model downloaded.")
        except Exception as e:
            st.error(f"Failed to download model: {e}")
            raise

    # load model
    try:
        model = joblib.load(LOCAL_MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        raise

# ---------- PREDICTION UTILITIES ----------
def prepare_dataframe_from_dict(row: Dict[str, Any]) -> pd.DataFrame:
    df = pd.DataFrame([row])
    for c in ALL_FEATURES:
        if c not in df.columns:
            df[c] = np.nan
    return df[ALL_FEATURES]

def append_inputs(df: pd.DataFrame):
    fp = os.path.join("inputs", "inputs.csv")
    if os.path.exists(fp):
        df.to_csv(fp, mode="a", index=False, header=False)
    else:
        df.to_csv(fp, index=False, header=True)

def predict_df(model, df: pd.DataFrame):
    # If pipeline supports predict_proba, get probability of positive class
    try:
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(df)
            # handle binary/multiclass
            pos_prob = probs[:, 1] if probs.shape[1] > 1 else probs[:, 0]
            preds = (pos_prob >= 0.5).astype(int)
            return pd.DataFrame({"prediction": preds, "probability": pos_prob})
        else:
            preds = model.predict(df)
            return pd.DataFrame({"prediction": preds, "probability": [None]*len(preds)})
    except Exception as e:
        st.error(f"Inference error: {e}")
        raise

# ---------- APP LAYOUT ----------
st.title("Wellness Tourism Package — Purchase Prediction")
st.markdown("Provide customer details (single record form) or upload a CSV with columns matching the training features.")

# load model lazily
model = None
try:
    model = download_and_load_model()
except Exception:
    st.warning("Model could not be downloaded/loaded. Please check HF_TOKEN and model repo/filename.")

# Single-record input form
with st.form("single_input"):
    st.subheader("Single record input")
    input_data = {}
    cols = st.columns(3)
    # numeric features in a grid
    for i, col_name in enumerate(NUMERIC_FEATURES):
        col = cols[i % 3]
        input_data[col_name] = col.number_input(label=col_name, value=0.0, format="%.2f")
    # categorical features (simple text inputs)
    for cat in CATEGORICAL_FEATURES:
        input_data[cat] = st.text_input(label=cat, value="")

    submitted = st.form_submit_button("Predict single record")
    if submitted:
        df_in = prepare_dataframe_from_dict(input_data)
        append_inputs(df_in)
        if model is not None:
            res_df = predict_df(model, df_in)
            st.write("### Input")
            st.dataframe(df_in)
            st.write("### Prediction")
            st.dataframe(res_df)
        else:
            st.error("Model not loaded.")

st.markdown("---")
# CSV upload flow
st.subheader("Batch prediction via CSV upload")
uploaded = st.file_uploader("Upload CSV (columns: " + ", ".join(ALL_FEATURES) + ")", type=["csv"])
if uploaded is not None:
    try:
        df_batch = pd.read_csv(uploaded)
        # ensure required columns exist
        for c in ALL_FEATURES:
            if c not in df_batch.columns:
                df_batch[c] = np.nan
        df_batch = df_batch[ALL_FEATURES]
        st.write("Uploaded sample:")
        st.dataframe(df_batch.head())

        if st.button("Run batch prediction"):
            append_inputs(df_batch)
            if model is not None:
                preds = predict_df(model, df_batch)
                out = pd.concat([df_batch.reset_index(drop=True), preds.reset_index(drop=True)], axis=1)
                out_fp = os.path.join("inputs", f"predictions_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv")
                out.to_csv(out_fp, index=False)
                st.success(f"Predictions saved to {out_fp}")
                st.dataframe(out.head())
            else:
                st.error("Model not loaded; cannot run prediction.")
    except Exception as e:
        st.error(f"Failed reading uploaded CSV: {e}")

st.markdown("---")
st.write("## Notes")
st.write("- The model is expected to be a scikit-learn Pipeline that accepts raw DataFrame rows using the feature names above.")
st.write("- Set HF_TOKEN in the environment or in the Space Secrets so the model can be downloaded.")
