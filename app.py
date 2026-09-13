import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler


st.set_page_config(page_title="Obesity Classification", page_icon="📊")
st.title("Lifestyle-Based Obesity Classification")
st.caption("Upload the project dataset once, then enter lifestyle information for a prediction.")

uploaded = st.file_uploader("Training dataset", type="csv")
if uploaded is None:
    st.info("Upload ObesityDataSet_raw_and_data_sinthetic.csv to initialise the model.")
    st.stop()


@st.cache_resource
def train_model(file_bytes):
    from io import BytesIO
    data = pd.read_csv(BytesIO(file_bytes)).drop_duplicates().copy()
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


model, features = train_model(uploaded.getvalue())
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

