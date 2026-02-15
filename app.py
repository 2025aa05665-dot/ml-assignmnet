import streamlit as st
import pandas as pd
import numpy as np
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report,
)

# PAGE CONFIG
st.set_page_config(page_title="Cardiovascular Dashboard", layout="wide")

st.title("Cardiovascular Disease Prediction Dashboard")

# SIDEBAR
st.sidebar.header("Controls")

model_name = st.sidebar.selectbox(
    "Select Model",
    [
        "logistic_regression",
        "decision_tree",
        "knn",
        "naive_bayes",
        "random_forest",
        "xgboost",
    ],
)

uploaded_file = st.sidebar.file_uploader(
    "Upload Test CSV (must include target column)", type=["csv"]
)

# SAMPLE TEST DATA DOWNLOAD
# SAMPLE TEST DATA GENERATION
@st.cache_data
def generate_sample_test():
    df = pd.read_csv("Cardiovascular_Disease_Dataset.csv")

    # drop patientid if present
    if "patientid" in df.columns:
        df = df.drop("patientid", axis=1)

    # take random 200 rows as test sample
    sample_df = df.sample(200, random_state=42)

    return sample_df


sample_df = generate_sample_test()

st.sidebar.download_button(
    label="⬇ Download Sample Test Data",
    data=sample_df.to_csv(index=False),
    file_name="sample_test.csv",
    mime="text/csv",
)

#LOAD MODEL
@st.cache_resource
def load_model(name):
    return joblib.load(f"model_files/{name}.pkl")



if uploaded_file is not None:

    data = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Data Preview")
    st.dataframe(data.head())

    if "target" not in data.columns:
        st.error("CSV must contain a 'target' column.")
        st.stop()

    X = data.drop("target", axis=1)
    y_true = data["target"]

    model = load_model(model_name)

    preds = model.predict(X)
    probs = model.predict_proba(X)[:, 1]

    # -------- METRICS --------
    acc = accuracy_score(y_true, preds)
    prec = precision_score(y_true, preds)
    rec = recall_score(y_true, preds)
    f1 = f1_score(y_true, preds)
    auc = roc_auc_score(y_true, probs)
    mcc = matthews_corrcoef(y_true, preds)

    # COLORED METRIC CARDS
    metric_colors = [
        "#16a34a",  # green - accuracy
        "#2563eb",  # blue - precision
        "#ea580c",  # orange - recall
        "#7c3aed",  # purple - f1
        "#059669",  # teal - auc
        "#4338ca",  # indigo - mcc
    ]

    metric_titles = ["Accuracy", "Precision", "Recall", "F1 Score", "ROC AUC", "MCC"]
    metric_values = [
        f"{acc:.2%}",
        f"{prec:.2f}",
        f"{rec:.2f}",
        f"{f1:.2f}",
        f"{auc:.2f}",
        f"{mcc:.2f}",
    ]

    cols = st.columns(6)

    for i in range(6):
        cols[i].markdown(
            f"""
            <div style="
                background-color:{metric_colors[i]};
                padding:18px;
                border-radius:14px;
                text-align:center;
                color:white;
                box-shadow:0 4px 12px rgba(0,0,0,0.25);
            ">
                <div style="font-size:14px; opacity:0.85;">{metric_titles[i]}</div>
                <div style="font-size:26px; font-weight:bold;">{metric_values[i]}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # -------- CONFUSION MATRIX --------
    st.subheader("Confusion Matrix")

    cm = confusion_matrix(y_true, preds)

    fig, ax = plt.subplots(figsize=(6, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="YlGn",
        cbar=True,
        linewidths=1,
        linecolor="white",
        xticklabels=["No Disease", "Disease"],
        yticklabels=["No Disease", "Disease"],
        annot_kws={"size": 14, "weight": "bold"},
        ax=ax,
    )

    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_title("Confusion Matrix", fontsize=14, weight="bold")

    st.pyplot(fig)

    #CLASSIFICATION REPORT#
    st.subheader("Classification Report")

    report = classification_report(y_true, preds, output_dict=True)
    report_df = pd.DataFrame(report).transpose()

    # Apply gradient to ALL numeric columns
    styled_report = (
        report_df.style
        .background_gradient(cmap="YlGn")   # color entire table
        .format("{:.3f}")                   # round numbers
        .set_properties(**{
            "text-align": "center",
            "font-size": "14px",
            "padding": "8px",
            "border": "1px solid #ddd"
        })
    )

    st.dataframe(styled_report, use_container_width=True)



else:
    st.info("⬅ Download sample test data from sidebar, then upload it to view predictions.")
