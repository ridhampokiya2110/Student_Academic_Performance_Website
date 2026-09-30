import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

st.set_page_config(
    page_title="Student Performance Predictor",
    layout="wide"
)

BASE = Path(__file__).parent
df = pd.read_csv(BASE / "student_performance_dataset.csv")
bundle = joblib.load(BASE / "model_bundle.pkl")
models = bundle["models"]
features = bundle["features"]
results = bundle["results"]
best_name = results.sort_values(
    ["R2_Score", "RMSE"], ascending=[False, True]
).iloc[0]["Model"]
best_model = models[best_name]

def category(marks):
    if marks < 40:
        return "At Risk"
    elif marks < 60:
        return "Average"
    elif marks < 75:
        return "Good"
    return "Excellent"

# ---------- Sidebar ----------
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "",
    ["Home", "Prediction", "Data Analysis", "Model Metrics", "About"]
)

# ---------- Home ----------
if page == "Home":
    st.title("Student Performance Dashboard")

    st.markdown("""
    This dashboard provides an interface to analyze student data and predict academic outcomes based on historical records.
    """)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Records", len(df))
    c2.metric("Features Used", len(features))
    c3.metric("Evaluated Models", len(models))
    c4.metric("Active Model", best_name)

    st.markdown("### Data Preview")
    st.dataframe(df.head(10), use_container_width=True)

# ---------- Prediction ----------
elif page == "Prediction":
    st.title("Performance Prediction")
    st.write("Enter student metrics below to estimate final marks.")

    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            study = st.slider("Study Hours/Day", 0.0, 12.0, 6.0, 0.5)
            attendance = st.slider("Attendance (%)", 0.0, 100.0, 85.0, 1.0)
            previous = st.slider("Previous Marks", 0.0, 100.0, 75.0, 1.0)

        with col2:
            assignment = st.slider("Assignment Score", 0.0, 100.0, 80.0, 1.0)
            internal = st.slider("Internal Marks", 0.0, 100.0, 78.0, 1.0)
            sleep = st.slider("Sleep Hours", 0.0, 12.0, 7.0, 0.5)

        with col3:
            failures = st.number_input("Previous Failures", 0, 5, 0, 1)
            internet = st.selectbox("Internet Access", ["Yes", "No"])
            extracurricular = st.selectbox(
                "Extracurricular Activities", ["Yes", "No"]
            )
            
        st.markdown("---")
        save_to_dataset = st.checkbox("Save record to dataset for future analysis")

        submitted = st.form_submit_button(
            "Run Prediction",
            use_container_width=True
        )

    if submitted:
        new_student = pd.DataFrame([{
            "Study_Hours": study,
            "Attendance": attendance,
            "Previous_Marks": previous,
            "Assignment_Score": assignment,
            "Internal_Marks": internal,
            "Sleep_Hours": sleep,
            "Failures": failures,
            "Internet_Access": 1 if internet == "Yes" else 0,
            "Extracurricular_Activities": 1 if extracurricular == "Yes" else 0
        }])

        prediction = float(best_model.predict(new_student[features])[0])
        prediction = float(np.clip(prediction, 0, 100))
        cat = category(prediction)

        st.markdown("---")
        a, b = st.columns(2)
        a.metric("Predicted Score", f"{prediction:.2f}")
        b.metric("Category", cat)

        if cat == "At Risk":
            st.warning("Assessment: Requires additional academic support.")
        elif cat == "Average":
            st.info("Assessment: Expected to meet baseline requirements.")
        elif cat == "Good":
            st.success("Assessment: On track for positive outcomes.")
        else:
            st.success("Assessment: Expected to perform exceptionally well.")

        if save_to_dataset:
            csv_path = BASE / "student_performance_dataset.csv"
            
            if not df.empty and "Student_ID" in df.columns:
                last_id = df["Student_ID"].iloc[-1]
                new_id_num = int(str(last_id).replace("S", "")) + 1
            else:
                new_id_num = 1
            new_student_id = f"S{new_id_num:03d}"
            
            new_record = pd.DataFrame([{
                "Student_ID": new_student_id,
                "Study_Hours": study,
                "Attendance": attendance,
                "Previous_Marks": previous,
                "Assignment_Score": assignment,
                "Internal_Marks": internal,
                "Sleep_Hours": sleep,
                "Failures": failures,
                "Internet_Access": internet,
                "Extracurricular_Activities": extracurricular,
                "Final_Marks": round(prediction, 1)
            }])
            
            new_record.to_csv(csv_path, mode='a', header=not csv_path.exists(), index=False)
            st.success(f"Record successfully saved to dataset with Student ID: {new_student_id}")

# ---------- EDA ----------
elif page == "Data Analysis":
    st.title("Exploratory Data Analysis")

    tab1, tab2, tab3, tab4 = st.tabs([
        "Distribution", "Study Hours", "Attendance", "Correlation"
    ])

    with tab1:
        fig, ax = plt.subplots(figsize=(9, 5))
        sns.histplot(df["Final_Marks"], bins=20, kde=True, ax=ax)
        ax.set_title("Distribution of Final Marks")
        st.pyplot(fig)

    with tab2:
        fig, ax = plt.subplots(figsize=(9, 5))
        sns.scatterplot(data=df, x="Study_Hours", y="Final_Marks", ax=ax)
        ax.set_title("Study Hours vs Final Marks")
        st.pyplot(fig)

    with tab3:
        fig, ax = plt.subplots(figsize=(9, 5))
        sns.scatterplot(data=df, x="Attendance", y="Final_Marks", ax=ax)
        ax.set_title("Attendance vs Final Marks")
        st.pyplot(fig)

    with tab4:
        numeric = df.select_dtypes(include=np.number)
        fig, ax = plt.subplots(figsize=(11, 7))
        sns.heatmap(numeric.corr(), annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
        ax.set_title("Feature Correlation Matrix")
        st.pyplot(fig)

    st.markdown("### Dataset Statistics")
    st.dataframe(df.describe().T, use_container_width=True)

# ---------- Model Comparison ----------
elif page == "Model Metrics":
    st.title("Model Performance Metrics")

    display_results = results.copy()
    display_results["MAE"] = display_results["MAE"].round(3)
    display_results["RMSE"] = display_results["RMSE"].round(3)
    display_results["R2_Score"] = display_results["R2_Score"].round(3)

    st.dataframe(display_results, use_container_width=True)

    c1, c2 = st.columns(2)

    with c1:
        fig, ax = plt.subplots(figsize=(8, 5))
        results.plot(
            x="Model", y=["MAE", "RMSE"], kind="bar", ax=ax
        )
        ax.set_title("Error Metrics by Model")
        ax.set_ylabel("Error Value")
        ax.tick_params(axis="x", rotation=0)
        st.pyplot(fig)

    with c2:
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.barplot(data=results, x="Model", y="R2_Score", ax=ax)
        ax.set_title("R² Score by Model")
        ax.set_ylim(0, 1)
        ax.tick_params(axis="x", rotation=0)
        st.pyplot(fig)

    st.info(f"Currently active model: {best_name}")

    if best_name == "Random Forest":
        importance = pd.DataFrame({
            "Feature": features,
            "Importance": best_model.feature_importances_
        }).sort_values("Importance", ascending=False)

        st.markdown("### Feature Importance")
        fig, ax = plt.subplots(figsize=(9, 6))
        sns.barplot(data=importance, x="Importance", y="Feature", ax=ax)
        st.pyplot(fig)

# ---------- About ----------
else:
    st.title("About")
    st.markdown("""
    This application evaluates academic records to model student outcomes. It uses standard regression techniques to identify risk factors and estimate final grades based on attendance, study habits, and historical scores.

    **Input Variables:**
    - Study Hours
    - Attendance Rate
    - Previous Academic Marks
    - Assignment Scores
    - Internal Assessment Marks
    - Sleep Duration
    - Previous Failures
    - Access to Internet
    - Extracurricular Participation

    **Output Variables:**
    - Estimated Final Score
    - Risk Category (At Risk, Average, Good, Excellent)
    """)
