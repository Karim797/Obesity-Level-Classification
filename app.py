import numpy as np
import pandas as pd
import streamlit as st
from io import BytesIO
from urllib.request import urlopen
from zipfile import ZipFile
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler


st.set_page_config(page_title="Obesity Classification", page_icon="📊")
st.title("Lifestyle-Based Obesity Classification")
st.caption("The official UCI dataset loads automatically. You can optionally upload a compatible CSV.")
st.warning("Educational demonstration only — this prediction is not medical advice or a clinical diagnosis.")

DATASET_URL = (
    "https://archive.ics.uci.edu/static/public/544/"
    "estimation%2Bof%2Bobesity%2Blevels%2Bbased%2Bon%2Beating%2Bhabits%2Band%2Bphysical%2Bcondition.zip"
)


@st.cache_data(show_spinner=False)
def load_uci_dataset():
    with urlopen(DATASET_URL, timeout=30) as response:
        archive = ZipFile(BytesIO(response.read()))
    csv_name = next(name for name in archive.namelist() if name.lower().endswith(".csv"))
    return pd.read_csv(archive.open(csv_name))


uploaded = st.file_uploader("Optional custom training dataset", type="csv")

try:
    data = pd.read_csv(uploaded) if uploaded is not None else load_uci_dataset()
except Exception as exc:
    st.error(f"The dataset could not be loaded: {exc}")
    st.stop()

st.success(
    f"Using {'your uploaded dataset' if uploaded is not None else 'the official UCI dataset'} "
    f"({len(data):,} rows)."
)


@st.cache_resource(show_spinner="Training the classification model...")
def train_model(csv_bytes):
    data = pd.read_csv(BytesIO(csv_bytes)).drop_duplicates().copy()
    target = "NObeyesdad"
    excluded = ["Height", "Weight"]
    X = data.drop(columns=[target] + excluded)
    y = data[target].astype(str)
    nominal = [c for c in ["Gender", "MTRANS"] if c in X]
    ordinal = [c for c in ["CAEC", "CALC"] if c in X]
    categorical = X.select_dtypes(include="object").columns.tolist()
    boolean = [c for c in categorical if c not in nominal + ordinal]
    numerical = X.select_dtypes(include=np.number).columns.tolist()
    prep = ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), numerical),
        ("nom", OneHotEncoder(handle_unknown="ignore"), nominal + boolean),
        ("ord", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), ordinal),
    ])
    model = Pipeline([("preprocess", prep), ("model", RandomForestClassifier(n_estimators=400, random_state=42))])
    model.fit(X, y)
    return model, X


model, features = train_model(data.to_csv(index=False).encode("utf-8"))
with st.form("prediction"):
    row = {}
    for column in features.columns:
        if pd.api.types.is_numeric_dtype(features[column]):
            row[column] = st.number_input(column, value=float(features[column].median()))
        else:
            row[column] = st.selectbox(column, sorted(features[column].astype(str).unique()))
    submitted = st.form_submit_button("Predict obesity level")

if submitted:
    sample = pd.DataFrame([row], columns=features.columns)
    prediction = model.predict(sample)[0]
    confidence = model.predict_proba(sample).max()
    st.success(f"Predicted class: {prediction}")
    st.metric("Confidence", f"{confidence:.1%}")
