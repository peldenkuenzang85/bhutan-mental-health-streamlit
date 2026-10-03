import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    precision_recall_fscore_support, roc_auc_score, roc_curve
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ============================================================
# CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Student Wellbeing Prediction",
    page_icon="🎓",
    layout="wide"
)

DATA_FILE = "synthetic_student_data.csv"
TARGET = "support_need"

NUMERIC_FEATURES = [
    "age", "sleep_hours", "study_hours_per_day",
    "physical_activity_days", "stress_level", "social_support",
    "financial_pressure", "academic_pressure", "screen_time_hours"
]

CATEGORICAL_FEATURES = [
    "gender", "district", "residence", "mental_health_awareness"
]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_FILE)
    if TARGET not in df.columns:
        raise ValueError(
            f"Dataset must contain '{TARGET}'. "
            f"Columns found: {list(df.columns)}"
        )
    return df

df = load_data()

missing = [c for c in FEATURES if c not in df.columns]
if missing:
    st.error("Missing required columns: " + ", ".join(missing))
    st.stop()

X = df[FEATURES].copy()
y = df[TARGET].copy()

# ============================================================
# TARGET PREPARATION
# ============================================================

if y.dtype == "object":
    y = y.astype(str).str.strip().str.lower().map({
        "0": 0, "1": 1,
        "no": 0, "yes": 1,
        "false": 0, "true": 1
    })

if y.isna().any() or y.nunique() != 2:
    st.error(
        "support_need must contain exactly two valid classes, "
        "such as 0/1 or Yes/No."
    )
    st.stop()

y = y.astype(int)

# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# ============================================================
# PREPROCESSING
# ============================================================

numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False
    ))
])

preprocessor = ColumnTransformer([
    ("num", numeric_transformer, NUMERIC_FEATURES),
    ("cat", categorical_transformer, CATEGORICAL_FEATURES)
])

# ============================================================
# LOGISTIC REGRESSION MODEL
# ============================================================

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(
        random_state=42,
        max_iter=1000,
        C=1.0
    ))
])

model.fit(X_train, y_train)

# ============================================================
# MODEL PREDICTIONS AND METRICS
# ============================================================

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_prob)

precision, recall, f1, _ = precision_recall_fscore_support(
    y_test,
    y_pred,
    average="binary",
    zero_division=0
)

# ============================================================
# PAGE TITLE AND TABS
# ============================================================

st.title("🎓 Student's Mental Health and Student Wellbeing in Bhutan")
st.caption("A Basic Machine-Learning Application Using Streamlit")

tab1, tab2, tab3, tab4 = st.tabs([
    "Overview",
    "Explore Data",
    "Prediction Demo",
    "Model Results"
])

# ============================================================
# TAB 1 — OVERVIEW
# ============================================================

with tab1:
    st.header("Project Overview")

    st.markdown("""
### Purpose
Predict whether a student may require support using
demographic, lifestyle, social, financial, academic and awareness factors.

### Preprocessing
- Missing numeric values → median imputation
- Missing categorical values → most-frequent imputation
- Numeric variables → StandardScaler
- Categorical variables → OneHotEncoder
- Unknown categories → ignored safely during prediction

### Model
- Algorithm: Logistic Regression
- Train/test split: 80% / 20%
- Random state: 42
- C = 1.0

### Responsible Use
This is a synthetic educational model. A prediction is not a medical or
psychological diagnosis and should not replace professional assessment.
""")

    a, b, c = st.columns(3)
    a.metric("Total Records", f"{len(df):,}")
    b.metric("Training Records", f"{len(X_train):,}")
    c.metric("Testing Records", f"{len(X_test):,}")

# ============================================================
# TAB 2 — EXPLORE DATA
# ============================================================

with tab2:
    st.header("Explore Dataset")
    a, b = st.columns(2)

    with a:
        st.write(f"Rows: **{df.shape[0]:,}**")
        st.write(f"Columns: **{df.shape[1]:,}**")
        st.write(
            f"Missing values: **{int(df.isna().sum().sum()):,}**"
        )

    with b:
        counts = y.value_counts().sort_index()
        fig, ax = plt.subplots(figsize=(6, 4))

        ax.bar(
            ["No Support (0)", "Support (1)"],
            [counts.get(0, 0), counts.get(1, 0)]
        )
        ax.set_ylabel("Number of Students")
        ax.set_title("Support Need Distribution")
        ax.tick_params(axis="x", rotation=10)

        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    feature = st.selectbox(
        "Select feature for distribution:",
        FEATURES
    )

    fig, ax = plt.subplots(figsize=(8, 4))

    if feature in NUMERIC_FEATURES:
        ax.hist(df[feature].dropna(), bins=20)
        ax.set_xlabel(feature)
        ax.set_ylabel("Frequency")
    else:
        counts = df[feature].fillna("Missing").value_counts()
        ax.bar(counts.index.astype(str), counts.values)
        ax.tick_params(axis="x", rotation=45)
        ax.set_xlabel(feature)
        ax.set_ylabel("Frequency")

    ax.set_title(f"Distribution of {feature}")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Sample Records")
    st.dataframe(df.head(20), use_container_width=True)

# ============================================================
# TAB 3 — PREDICTION DEMO
# ============================================================

with tab3:
    st.header("Prediction Demo")
    st.write("Enter a student's information and predict support need.")

    with st.form("prediction_form"):
        c1, c2, c3 = st.columns(3)

        with c1:
            age = st.number_input("Age", 16, 25, 20, 1)
            gender = st.selectbox(
                "Gender",
                ["Male", "Female", "Other"]
            )

            districts = sorted(
                df["district"].dropna().astype(str).unique()
            )

            district = st.selectbox("District", districts)
            residence = st.selectbox(
                "Residence",
                ["Urban", "Rural"]
            )

        with c2:
            sleep = st.number_input(
                "Sleep Hours",
                4.0, 10.0, 7.0, 0.5
            )
            study = st.number_input(
                "Study Hours/Day",
                1.0, 10.0, 4.0, 0.5
            )
            activity = st.number_input(
                "Physical Activity Days/Week",
                0, 7, 3, 1
            )
            stress = st.slider(
                "Stress Level",
                1, 10, 5
            )

        with c3:
            support = st.slider(
                "Social Support",
                1, 10, 5
            )
            financial = st.slider(
                "Financial Pressure",
                1, 10, 5
            )
            academic = st.slider(
                "Academic Pressure",
                1, 10, 5
            )
            screen = st.number_input(
                "Screen Time Hours",
                1.0, 12.0, 5.0, 0.5
            )

        awareness_values = sorted(
            df["mental_health_awareness"]
            .dropna()
            .astype(str)
            .unique()
        )

        awareness = st.selectbox(
            "Mental Health Awareness",
            awareness_values
        )

        submitted = st.form_submit_button(
            "Predict Support Need",
            type="primary"
        )

    if submitted:
        input_df = pd.DataFrame({
            "age": [age],
            "gender": [gender],
            "district": [district],
            "residence": [residence],
            "sleep_hours": [sleep],
            "study_hours_per_day": [study],
            "physical_activity_days": [activity],
            "stress_level": [stress],
            "social_support": [support],
            "financial_pressure": [financial],
            "academic_pressure": [academic],
            "screen_time_hours": [screen],
            "mental_health_awareness": [awareness]
        })

        prediction = int(model.predict(input_df)[0])
        probability = float(
            model.predict_proba(input_df)[0, 1]
        )

        st.subheader("Prediction Result")
        r1, r2 = st.columns(2)

        with r1:
            if prediction == 1:
                st.warning(
                    "Predicted Support Need: **YES (1)**"
                )
            else:
                st.success(
                    "Predicted Support Need: **NO (0)**"
                )

        with r2:
            st.metric(
                "Probability of Support Need",
                f"{probability:.1%}"
            )

        st.progress(probability)

# ============================================================
# TAB 4 — MODEL RESULTS
# ============================================================

with tab4:
    st.header("Logistic Regression Model Results")

    # --------------------------------------------------------
    # MODEL METRICS
    # --------------------------------------------------------

    c1, c2, c3, c4, c5 = st.columns(5)

    c1.metric("Accuracy", f"{accuracy:.4f}")
    c2.metric("ROC-AUC", f"{roc_auc:.4f}")
    c3.metric("Precision", f"{precision:.4f}")
    c4.metric("Recall", f"{recall:.4f}")
    c5.metric("F1-Score", f"{f1:.4f}")

    # --------------------------------------------------------
    # CONFUSION MATRIX
    # --------------------------------------------------------

    st.subheader("Confusion Matrix")

    cm = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(6, 4))
    image = ax.imshow(cm)

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])

    ax.set_xticklabels(
        ["Predicted 0", "Predicted 1"],
        fontsize=10
    )
    ax.set_yticklabels(
        ["Actual 0", "Actual 1"],
        fontsize=10
    )

    ax.set_xlabel("Predicted", fontsize=10)
    ax.set_ylabel("Actual", fontsize=10)
    ax.set_title("Confusion Matrix", fontsize=12)

    for i in range(2):
        for j in range(2):
            ax.text(
                j, i, str(cm[i, j]),
                ha="center",
                va="center",
                fontsize=13
            )

    fig.tight_layout()
    st.pyplot(fig, use_container_width=False)
    plt.close(fig)

    # --------------------------------------------------------
    # ROC CURVE
    # --------------------------------------------------------

    st.subheader("ROC Curve")

    fpr, tpr, _ = roc_curve(y_test, y_prob)

    fig, ax = plt.subplots(figsize=(6, 4))

    ax.plot(
        fpr,
        tpr,
        label=f"Logistic Regression (AUC = {roc_auc:.4f})"
    )
    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--"
    )

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend()

    fig.tight_layout()
    st.pyplot(fig, use_container_width=False)
    plt.close(fig)

    # --------------------------------------------------------
    # CLASSIFICATION REPORT
    # --------------------------------------------------------

    st.subheader("Classification Report")

    report = classification_report(
        y_test,
        y_pred,
        target_names=[
            "No Support Needed",
            "Support Needed"
        ],
        output_dict=True,
        zero_division=0
    )

    report_df = pd.DataFrame(report).transpose()

    report_df = report_df.rename(index={
        "No Support Needed": "No Support Needed",
        "Support Needed": "Support Needed",
        "accuracy": "Accuracy",
        "macro avg": "Macro Average",
        "weighted avg": "Weighted Average"
    })

    report_df = report_df.rename(columns={
        "precision": "Precision",
        "recall": "Recall",
        "f1-score": "F1-Score",
        "support": "Support"
    })

    report_df["Support"] = report_df["Support"].astype(int)

    st.dataframe(
        report_df.style.format({
            "Precision": "{:.2f}",
            "Recall": "{:.2f}",
            "F1-Score": "{:.2f}",
            "Support": "{:.0f}"
        }),
        use_container_width=False,
        hide_index=False
    )

    # --------------------------------------------------------
    # MODEL CONFIGURATION
    # --------------------------------------------------------

    st.subheader("Model Configuration")

    st.json({
        "algorithm": "Logistic Regression",
        "C": 1.0,
        "max_iter": 1000,
        "test_size": 0.20,
        "random_state": 42,
        "numeric_imputation": "median",
        "categorical_imputation": "most_frequent",
        "numeric_scaling": "StandardScaler",
        "categorical_encoding": "OneHotEncoder(handle_unknown='ignore')"
    })