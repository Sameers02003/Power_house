import streamlit as st
import sqlite3
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# ---- DATABASE CONNECTION ----
@st.cache_data
def load_data():
    conn = sqlite3.connect("power_pulse.db")
    query = "SELECT * FROM power_consumption ORDER BY Datetime DESC LIMIT 10000"
    df = pd.read_sql(query, conn)
    conn.close()

    # Convert 'Datetime' column properly
    df["Datetime"] = pd.to_datetime(df["Datetime"], errors="coerce")

    # Convert numeric columns for proper aggregation
    numeric_cols = ["Global_active_power", "Global_reactive_power", "Voltage",
                    "Global_intensity", "Sub_metering_1", "Sub_metering_2", "Sub_metering_3"]

    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")  # Convert to numeric safely

    # Aggregate to hourly intervals
    df = df.set_index("Datetime").resample("H")[numeric_cols].mean().reset_index()

    return df

df = load_data()

# ---- NAVIGATION ----
page = st.sidebar.selectbox("Select Page", ["Dashboard", "Project Summary"])

# ---- PROJECT SUMMARY PAGE ----
if page == "Project Summary":
    st.title("📌 Project Summary")
    
    st.write("### 🔹 Project Name: PowerPulse - Household Energy Usage Forecast")
    st.write("This project aims to analyze household power consumption data and build predictive models to optimize energy usage.")
    
    st.write("### 🔹 Key Objectives")
    st.markdown("- Perform **Exploratory Data Analysis (EDA)** to uncover patterns in energy consumption.")
    st.markdown("- Develop **feature engineering techniques** to improve prediction accuracy.")
    st.markdown("- Build and evaluate **predictive models** using regression and time-series analysis.")
    
    st.write("### 🔹 Data Overview")
    st.write("The dataset contains power consumption metrics, including active power, voltage fluctuations, and appliance-level usage.")

    st.write("### 🔹 Evaluation Metrics")
    st.markdown("- **Root Mean Squared Error (RMSE):** Evaluates prediction errors.")
    st.markdown("- **Mean Absolute Error (MAE):** Measures the average magnitude of prediction errors.")
    st.markdown("- **R² Score:** Assesses how well the model explains variations in power consumption.")

    st.write("### 🔹 Conclusion")
    st.write("This project provides a **data-driven approach** to forecasting energy consumption, enabling users to optimize power usage and reduce costs.")

# ---- DASHBOARD PAGE ----
if page == "Dashboard":
    st.title("⚡ Household Power Consumption Dashboard")
    st.write("Explore trends, patterns, and distribution in power usage data.")

    # ---- DATE FILTER ----
    st.sidebar.write("## Filter Data")
    start_date = st.sidebar.date_input("Start Date", df["Datetime"].min().date())
    end_date = st.sidebar.date_input("End Date", df["Datetime"].max().date())

    # Filter data based on user input
    filtered_df = df[(df["Datetime"] >= str(start_date)) & (df["Datetime"] <= str(end_date))]

    # ---- DISPLAY FILTERED DATA ----
    if not filtered_df.empty:
        st.write("### Filtered Data View")
        st.dataframe(filtered_df)
    else:
        st.warning("No data available for the selected date range. Adjust your filter.")

    # ---- VISUALIZATIONS ----
    st.write("## 🔹 Power Consumption Trends")
    if not filtered_df.empty:
        st.line_chart(filtered_df[["Datetime", "Global_active_power"]], x="Datetime", y="Global_active_power")
    else:
        st.warning("No data available for visualization.")

    st.write("## 🔹 Voltage Distribution")
    if not filtered_df.empty:
        st.bar_chart(filtered_df["Voltage"])
    else:
        st.warning("No data available for visualization.")

    st.write("## 🔹 Sub-Metering Usage Over Time")
    if not filtered_df.empty:
        st.line_chart(filtered_df[["Datetime", "Sub_metering_1", "Sub_metering_2", "Sub_metering_3"]], x="Datetime")
    else:
        st.warning("No data available for visualization.")

    # ---- FEATURE IMPORTANCE ANALYSIS ----
    st.write("## 🔹 Feature Importance - Correlation Heatmap")
    if not filtered_df.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(filtered_df.drop(columns=["Datetime"]).corr(), annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)
    else:
        st.warning("No data available for feature correlation analysis.")

    # ---- MODEL PERFORMANCE ----
    st.write("## 🔹 Model Evaluation Metrics")
    if not filtered_df.empty:
        # Placeholder values (Replace with actual calculated metrics)
        st.write("**RMSE:** 0.245")
        st.write("**MAE:** 0.180")
        st.write("**R² Score:** 0.92")
    else:
        st.warning("No data available for model evaluation.")

    # ---- ACTUAL VS PREDICTED VISUALIZATION ----
    st.write("## 🔹 Predicted vs. Actual Power Consumption")
    if not filtered_df.empty:
        # Placeholder prediction logic (Replace with actual model predictions)
        filtered_df["Predicted"] = filtered_df["Global_active_power"] * 1.02  # Example adjustment
        st.line_chart(filtered_df[["Datetime", "Global_active_power", "Predicted"]], x="Datetime")
    else:
        st.warning("No data available for predictions.")
