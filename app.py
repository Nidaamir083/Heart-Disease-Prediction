import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, roc_auc_score, accuracy_score

st.set_page_config(page_title="Heart Disease Dashboard", layout="wide")

@st.cache_data
def load_data(path="heart.csv"):
    try:
        df = pd.read_csv(path)
        st.success(f"Loaded dataset from {path}")
    except FileNotFoundError:
        st.warning(f"'{path}' not found. Using synthetic fallback sample.")
        # minimal synthetic example consistent with expected columns
        df = pd.DataFrame({
            "age": [63, 67, 67, 37],
            "sex": [1, 1, 1, 1],
            "cp": [3, 2, 4, 3],
            "trestbps": [145, 160, 120, 130],
            "chol": [233, 286, 229, 250],
            "fbs": [1, 0, 0, 0],
            "restecg": [0, 0, 0, 1],
            "thalach": [150, 108, 129, 187],
            "exang": [0, 1, 1, 0],
            "oldpeak": [2.3, 1.5, 2.6, 3.5],
            "slope": [0, 2, 1, 0],
            "ca": [0, 3, 2, 0],
            "thal": [1, 2, 2, 3],
            "target": [1, 0, 1, 0]
        })
    return df

def build_models():
    return {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000))
        ]),
        "Random Forest": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(n_estimators=100, random_state=42))
        ])
    }

# Load
df = load_data()

st.title("Heart Disease Analysis & Prediction")
st.markdown("Dataset: UCI Heart Disease. Target=1 indicates presence of disease.")

# Sidebar controls
st.sidebar.header("Configuration")
show_data = st.sidebar.checkbox("Show raw data", value=False)
model_choice = st.sidebar.selectbox("Select model", ["Logistic Regression", "Random Forest"])
test_size = st.sidebar.slider("Test set fraction", 0.1, 0.4, 0.2)
random_state = st.sidebar.number_input("Random seed", value=42, step=1)

# Override default model via secrets if provided
default_model_secret = st.secrets.get("app", {}).get("default_model", None)
if default_model_secret and default_model_secret in ["Logistic Regression", "Random Forest"]:
    model_choice = default_model_secret

if show_data:
    st.subheader("Raw Data")
    st.dataframe(df)

# Preprocess
if "target" not in df.columns:
    st.error("Dataset missing 'target' column. Cannot proceed.")
    st.stop()

X = df.drop(columns=["target"])
y = df["target"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=random_state, stratify=y if len(y.unique()) > 1 else None
)

# Train selected model
model = build_models()[model_choice]
model.fit(X_train, y_train)
preds = model.predict(X_test)
probs = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

# Metrics display
st.subheader("Model Evaluation")
col1, col2 = st.columns(2)
with col1:
    st.metric("Accuracy", f"{accuracy_score(y_test, preds):.3f}")
    if probs is not None:
        st.metric("ROC AUC", f"{roc_auc_score(y_test, probs):.3f}")
with col2:
    st.write("Confusion Matrix")
    cm = confusion_matrix(y_test, preds)
    st.write(cm)

# Feature distribution
st.subheader("Feature Distributions")
feature = st.selectbox("Feature", X.columns.tolist())
fig, ax = plt.subplots()
for target_val in sorted(df["target"].unique()):
    subset = df[df["target"] == target_val]
    ax.hist(subset[feature], bins=20, alpha=0.5, label=f"target={target_val}")
ax.set_xlabel(feature)
ax.set_ylabel("Count")
ax.legend()
st.pyplot(fig)

# Individual prediction
st.subheader("Individual Prediction")
st.write("Input patient features to get prediction")
input_data = {}
for col in X.columns:
    default = float(df[col].median()) if pd.api.types.is_numeric_dtype(df[col]) else 0.0
    input_data[col] = st.number_input(col, value=default, format="%.3f")
input_df = pd.DataFrame([input_data])
prediction = model.predict(input_df)[0]
prob_display = ""
if hasattr(model, "predict_proba"):
    prob = model.predict_proba(input_df)[0][1]
    prob_display = f"Probability of disease: **{prob:.3f}**"
st.markdown(f"**Predicted target:** {int(prediction)} (1 = disease)")
if prob_display:
    st.markdown(prob_display)


