import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import glob
from sklearn.metrics import (
    confusion_matrix,
    classification_report
)

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Breast Cancer Classification",
    page_icon="🩺",
    layout="wide"
)

# ============================================================
# TITLE
# ============================================================

st.title("🩺 Breast Cancer Classification")
st.subheader(
    "Machine Learning Classification using the Breast Cancer (UCI) Dataset"
)

st.markdown("""
This application demonstrates five machine learning classification models
trained on the Breast Cancer (UCI) dataset.
""")

# ============================================================
# FILE PATHS
# ============================================================

MODEL_FOLDER = "models"
TEST_FILE = "test_data.csv"

# ============================================================
# LOAD SCALER
# ============================================================

try:
    scaler = joblib.load(
        os.path.join(MODEL_FOLDER, "scaler.pkl")
    )
except Exception as e:
    st.error(f"Unable to load scaler.pkl: {e}")
    st.stop()

# ============================================================
# FIND SAVED MODELS AUTOMATICALLY
# ============================================================

model_files = glob.glob(
    os.path.join(MODEL_FOLDER, "*.pkl")
)


def find_model(keyword):
    for file in model_files:
        filename = os.path.basename(file).lower()

        if keyword in filename and "scaler" not in filename:
            return file

    return None


model_paths = {
    "Logistic Regression": find_model("logistic"),
    "Decision Tree": find_model("decision"),
    "kNN": find_model("knn"),
    "Naive Bayes": find_model("naive"),
    "Random Forest": find_model("random")
}

models = {}

for model_name, model_path in model_paths.items():

    if model_path is not None:

        try:
            models[model_name] = joblib.load(model_path)

        except Exception as e:
            st.warning(
                f"Could not load {model_name}: {e}"
            )

# ============================================================
# LOAD DEFAULT TEST DATA
# ============================================================

try:
    default_test_data = pd.read_csv(TEST_FILE)

except Exception as e:
    st.error(
        f"Unable to load test_data.csv: {e}"
    )
    st.stop()

# ============================================================
# TARGET AND FEATURES
# ============================================================

TARGET = "benign_0__mal_1"

# ============================================================
# MODEL PERFORMANCE
# ============================================================

performance = pd.DataFrame({

    "Model": [
        "Logistic Regression",
        "Decision Tree",
        "kNN",
        "Naive Bayes",
        "Random Forest"
    ],

    "Accuracy": [
        0.9825,
        0.9123,
        0.9561,
        0.9298,
        0.9561
    ],

    "AUC": [
        0.9954,
        0.9157,
        0.9788,
        0.9868,
        0.9937
    ],

    "Precision": [
        0.9861,
        0.9559,
        0.9589,
        0.9444,
        0.9589
    ],

    "Recall": [
        0.9861,
        0.9028,
        0.9722,
        0.9444,
        0.9722
    ],

    "F1 Score": [
        0.9861,
        0.9286,
        0.9655,
        0.9444,
        0.9655
    ],

    "MCC": [
        0.9623,
        0.8174,
        0.9054,
        0.8492,
        0.9054
    ]
})

# ============================================================
# SIDEBAR NAVIGATION
# ============================================================

st.sidebar.header("Navigation")

page = st.sidebar.radio(
    "Select Page",
    [
        "Dataset Overview",
        "Model Performance",
        "Prediction"
    ]
)

# ============================================================
# PAGE 1 - DATASET OVERVIEW
# ============================================================

if page == "Dataset Overview":

    st.header("📊 Dataset Overview")

    # --------------------------------------------------------
    # CSV UPLOAD
    # --------------------------------------------------------

    st.subheader("Upload Test Data")

    uploaded_file = st.file_uploader(
        "Upload Test Data (CSV)",
        type=["csv"],
        help="Upload the test dataset in CSV format."
    )

    if uploaded_file is not None:

        try:

            uploaded_data = pd.read_csv(
                uploaded_file
            )

            if TARGET not in uploaded_data.columns:

                st.error(
                    f"The uploaded CSV must contain the target column: "
                    f"{TARGET}"
                )

            else:

                st.session_state["test_data"] = uploaded_data

                st.success(
                    "Test data uploaded successfully."
                )

        except Exception as e:

            st.error(
                f"Unable to read uploaded CSV: {e}"
            )

    # --------------------------------------------------------
    # USE UPLOADED DATA OR DEFAULT DATA
    # --------------------------------------------------------

    test_data = st.session_state.get(
        "test_data",
        default_test_data
    )

    FEATURES = [
        column
        for column in test_data.columns
        if column != TARGET
    ]

    # --------------------------------------------------------
    # DATASET INFORMATION
    # --------------------------------------------------------

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Total Samples",
            "569"
        )

    with col2:
        st.metric(
            "Features",
            "30"
        )

    with col3:
        st.metric(
            "Training Samples",
            "455"
        )

    with col4:
        st.metric(
            "Testing Samples",
            len(test_data)
        )

    st.divider()

    # --------------------------------------------------------
    # TARGET VARIABLE
    # --------------------------------------------------------

    st.subheader("Target Variable")

    target_counts = pd.DataFrame({

        "Class": [
            "Benign (0)",
            "Malignant (1)"
        ],

        "Count": [
            int(
                (test_data[TARGET] == 0).sum()
            ),

            int(
                (test_data[TARGET] == 1).sum()
            )
        ]
    })

    st.dataframe(
        target_counts,
        use_container_width=True,
        hide_index=True
    )

    # --------------------------------------------------------
    # DATASET SAMPLE
    # --------------------------------------------------------

    st.subheader("Dataset Sample")

    st.dataframe(
        test_data.head(10),
        use_container_width=True
    )

    st.info(
        "Target encoding: 0 = Benign, 1 = Malignant"
    )

# ============================================================
# PAGE 2 - MODEL PERFORMANCE
# ============================================================

elif page == "Model Performance":

    st.header("🤖 Model Performance Comparison")

    # --------------------------------------------------------
    # COMPARISON TABLE
    # --------------------------------------------------------

    st.dataframe(

        performance.style.format({

            "Accuracy": "{:.4f}",
            "AUC": "{:.4f}",
            "Precision": "{:.4f}",
            "Recall": "{:.4f}",
            "F1 Score": "{:.4f}",
            "MCC": "{:.4f}"

        }),

        use_container_width=True,
        hide_index=True
    )

    st.divider()

    # --------------------------------------------------------
    # BEST MODEL
    # --------------------------------------------------------

    st.subheader("Best Performing Model")

    best_model = performance.loc[
        performance["Accuracy"].idxmax()
    ]

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Best Model",
            best_model["Model"]
        )

    with col2:

        st.metric(
            "Accuracy",
            f"{best_model['Accuracy']:.4f}"
        )

    with col3:

        st.metric(
            "AUC",
            f"{best_model['AUC']:.4f}"
        )

    st.success(
        "Logistic Regression achieved the best overall performance "
        "with an Accuracy of 98.25% and AUC of 99.54%."
    )

    # --------------------------------------------------------
    # CONFUSION MATRIX / CLASSIFICATION REPORT
    # --------------------------------------------------------

    st.divider()

    st.subheader(
        "Confusion Matrix / Classification Report"
    )

    selected_performance_model = st.selectbox(
        "Select Model for Test Data Evaluation",
        list(models.keys())
    )

    test_data = st.session_state.get(
        "test_data",
        default_test_data
    )

    FEATURES = [
        column
        for column in test_data.columns
        if column != TARGET
    ]

    X_eval = test_data[FEATURES]
    y_eval = test_data[TARGET]

    selected_eval_model = models[
        selected_performance_model
    ]

    try:

        X_eval_scaled = scaler.transform(
            X_eval
        )

        y_pred = selected_eval_model.predict(
            X_eval_scaled
        )

        # ----------------------------------------------------
        # CONFUSION MATRIX
        # ----------------------------------------------------

        st.write(
            f"**Confusion Matrix - "
            f"{selected_performance_model}**"
        )

        cm = confusion_matrix(
            y_eval,
            y_pred
        )

        cm_df = pd.DataFrame(
            cm,
            index=[
                "Actual Benign",
                "Actual Malignant"
            ],
            columns=[
                "Predicted Benign",
                "Predicted Malignant"
            ]
        )

        st.dataframe(
            cm_df,
            use_container_width=True
        )

        # ----------------------------------------------------
        # CLASSIFICATION REPORT
        # ----------------------------------------------------

        st.write(
            f"**Classification Report - "
            f"{selected_performance_model}**"
        )

        report = classification_report(
            y_eval,
            y_pred,
            target_names=[
                "Benign",
                "Malignant"
            ],
            output_dict=True
        )

        report_df = pd.DataFrame(
            report
        ).transpose()

        st.dataframe(
            report_df.style.format(
                "{:.4f}"
            ),
            use_container_width=True
        )

    except Exception as e:

        st.error(
            f"Unable to generate evaluation results: {e}"
        )

# ============================================================
# PAGE 3 - PREDICTION
# ============================================================

elif page == "Prediction":

    st.header("🔮 Breast Cancer Prediction")

    st.write(
        "Select a machine learning model and a sample "
        "from the test dataset."
    )

    # --------------------------------------------------------
    # GET CURRENT TEST DATA
    # --------------------------------------------------------

    test_data = st.session_state.get(
        "test_data",
        default_test_data
    )

    FEATURES = [
        column
        for column in test_data.columns
        if column != TARGET
    ]

    X_test = test_data[FEATURES]

    y_test = test_data[TARGET]

    # --------------------------------------------------------
    # MODEL SELECTION
    # --------------------------------------------------------

    available_models = list(
        models.keys()
    )

    if len(available_models) == 0:

        st.error(
            "No trained model files were found "
            "in the models folder."
        )

        st.write(
            "Please check that your .pkl model files "
            "are inside:"
        )

        st.code("models/")

        st.stop()

    selected_model_name = st.selectbox(
        "Select Machine Learning Model",
        available_models
    )

    selected_model = models[
        selected_model_name
    ]

    # --------------------------------------------------------
    # SAMPLE SELECTION
    # --------------------------------------------------------

    sample_number = st.number_input(
        "Select Test Sample Number",
        min_value=0,
        max_value=len(X_test) - 1,
        value=0,
        step=1
    )

    sample = X_test.iloc[
        [sample_number]
    ]

    actual_value = int(
        y_test.iloc[sample_number]
    )

    # --------------------------------------------------------
    # DISPLAY INPUT
    # --------------------------------------------------------

    st.subheader(
        "Selected Patient Features"
    )

    st.dataframe(
        sample,
        use_container_width=True
    )

    # --------------------------------------------------------
    # PREDICTION
    # --------------------------------------------------------

    if st.button("🔍 Predict"):

        try:

            # Scale input using same scaler
            sample_scaled = scaler.transform(
                sample
            )

            prediction = selected_model.predict(
                sample_scaled
            )[0]

            # ------------------------------------------------
            # PROBABILITY
            # ------------------------------------------------

            probability = None

            if hasattr(
                selected_model,
                "predict_proba"
            ):

                probabilities = (
                    selected_model.predict_proba(
                        sample_scaled
                    )
                )

                probability = (
                    probabilities[0][1]
                )

            # ------------------------------------------------
            # RESULTS
            # ------------------------------------------------

            st.subheader(
                "Prediction Result"
            )

            col1, col2, col3 = st.columns(3)

            with col1:

                if prediction == 0:

                    st.success(
                        "Prediction: BENIGN"
                    )

                else:

                    st.error(
                        "Prediction: MALIGNANT"
                    )

            with col2:

                if actual_value == 0:

                    st.info(
                        "Actual: BENIGN"
                    )

                else:

                    st.warning(
                        "Actual: MALIGNANT"
                    )

            with col3:

                if probability is not None:

                    st.metric(
                        "Malignant Probability",
                        f"{probability * 100:.2f}%"
                    )

            # ------------------------------------------------
            # CORRECT / INCORRECT
            # ------------------------------------------------

            if prediction == actual_value:

                st.success(
                    "✅ Prediction is CORRECT"
                )

            else:

                st.error(
                    "❌ Prediction is INCORRECT"
                )

        except Exception as e:

            st.error(
                f"Prediction error: {e}"
            )

# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Breast Cancer Classification | Machine Learning Assignment"
)
