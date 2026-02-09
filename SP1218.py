import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# -------------------------------
# APP TITLE
# -------------------------------
st.title("💼 Salary Prediction using Random Forest")

# -------------------------------
# LOAD DATA
# -------------------------------
df = pd.read_csv("salary_dataset_2000.csv")
st.success("Dataset Loaded Successfully")

st.write("Dataset Preview:")
st.dataframe(df.head())

# -------------------------------
TARGET = "Salary"

X = df.drop(TARGET, axis=1)
y = df[TARGET]

num_features = X.select_dtypes(include=["int64", "float64"]).columns
cat_features = X.select_dtypes(include=["object"]).columns

# -------------------------------
# PREPROCESSING
# -------------------------------
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, num_features),
        ("cat", categorical_transformer, cat_features)
    ]
)

# -------------------------------
# TRAIN TEST SPLIT
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# remove NaN target
train_mask = ~y_train.isna()
test_mask = ~y_test.isna()

X_train = X_train[train_mask]
y_train = y_train[train_mask]

X_test = X_test[test_mask]
y_test = y_test[test_mask]

# -------------------------------
# MODEL
# -------------------------------
rf = RandomForestRegressor(
    n_estimators=400,
    max_depth=15,
    min_samples_split=5,
    random_state=42,
    n_jobs=-1
)

pipe = Pipeline(steps=[
    ("preprocess", preprocessor),
    ("model", rf)
])

# -------------------------------
# TRAIN
# -------------------------------
with st.spinner("Training model..."):
    pipe.fit(X_train, y_train)

st.success("Model Trained!")

# -------------------------------
# METRICS
# -------------------------------
y_pred = pipe.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

n = X_test.shape[0]
p = X_test.shape[1]
adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)

st.subheader("📊 Model Performance")

st.write("MAE:", mae)
st.write("RMSE:", rmse)
st.write("R²:", r2)
st.write("Adjusted R²:", adj_r2)

# -------------------------------
# VISUALS
# -------------------------------
st.subheader("📈 Training vs Testing Visualizations")

# Training plot
fig1 = plt.figure()
plt.scatter(y_train, pipe.predict(X_train))
plt.xlabel("Actual Salary (Train)")
plt.ylabel("Predicted Salary (Train)")
plt.title("Training: Actual vs Predicted")
st.pyplot(fig1)

# Testing plot
fig2 = plt.figure()
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Salary (Test)")
plt.ylabel("Predicted Salary (Test)")
plt.title("Testing: Actual vs Predicted")
st.pyplot(fig2)

# Residual plot
fig3 = plt.figure()
plt.hist(y_test - y_pred, bins=30)
plt.xlabel("Residual")
plt.title("Residual Distribution")
st.pyplot(fig3)


