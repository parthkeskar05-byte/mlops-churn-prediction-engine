"""Streamlit dashboard for real-time churn monitoring."""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import yaml
import requests
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parents[2]))

from src.data.loader import load_data
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

# Page config
st.set_page_config(
    page_title="Churn Prediction Dashboard",
    page_icon="📊",
    layout="wide"
)

# Load configuration
@st.cache_resource
def load_config():
    config_path = Path(__file__).parents[2] / "config" / "config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)

config = load_config()

# Title
st.title("🎯 Churn Prediction Monitoring Dashboard")
st.markdown("Real-time customer churn prediction monitoring and analysis")

# Sidebar
st.sidebar.header("Configuration")
api_url = st.sidebar.text_input(
    "API Endpoint",
    value=f"http://localhost:{config['api']['port']}"
)

# Main content
tab1, tab2, tab3 = st.tabs(["📊 Model Metrics", "🔮 Make Prediction", "📈 Feature Importance"])

# Tab 1: Model Metrics
with tab1:
    st.header("Model Performance Metrics")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("API Health Check")
        try:
            response = requests.get(f"{api_url}/health", timeout=5)
            if response.status_code == 200:
                health_data = response.json()
                if health_data["model_loaded"]:
                    st.success("✅ Model is loaded and ready")
                else:
                    st.warning("⚠️ Model not loaded")
            else:
                st.error("❌ API not responding")
        except Exception as e:
            st.error(f"❌ Cannot connect to API: {e}")

    with col2:
        st.subheader("API Metrics")
        try:
            response = requests.get(f"{api_url}/metrics", timeout=5)
            if response.status_code == 200:
                metrics_data = response.json()
                st.metric("Total Predictions", metrics_data["total_predictions"])
                st.metric("Avg Latency (ms)", f"{metrics_data['avg_latency_ms']:.2f}")
            else:
                st.info("No metrics available yet")
        except Exception:
            st.info("Metrics unavailable")

    # Display sample data statistics
    st.subheader("Dataset Statistics")
    try:
        data_path = Path(__file__).parents[2] / config['paths']['data_path']
        if data_path.exists():
            df = load_data(str(data_path))

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Customers", len(df))
            with col2:
                churn_rate = df['Churn'].mean() * 100
                st.metric("Churn Rate", f"{churn_rate:.1f}%")
            with col3:
                avg_tenure = df['tenure'].mean()
                st.metric("Avg Tenure (months)", f"{avg_tenure:.1f}")
            with col4:
                avg_charges = df['MonthlyCharges'].mean()
                st.metric("Avg Monthly Charges", f"${avg_charges:.2f}")

            # Churn distribution
            st.subheader("Churn Distribution")
            churn_counts = df['Churn'].value_counts()
            fig = px.pie(
                values=churn_counts.values,
                names=['No Churn', 'Churn'],
                title="Customer Churn Distribution",
                color_discrete_sequence=['#00cc96', '#ef553b']
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Dataset not found. Generate sample data first.")
    except Exception as e:
        st.error(f"Error loading data: {e}")

# Tab 2: Make Prediction
with tab2:
    st.header("Predict Customer Churn")

    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            gender = st.selectbox("Gender", ["Male", "Female"])
            senior_citizen = st.selectbox("Senior Citizen", [0, 1])
            partner = st.selectbox("Partner", ["Yes", "No"])
            dependents = st.selectbox("Dependents", ["Yes", "No"])
            tenure = st.number_input("Tenure (months)", 0, 72, 12)
            phone_service = st.selectbox("Phone Service", ["Yes", "No"])

        with col2:
            multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
            internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
            online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
            online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
            device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
            tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])

        with col3:
            streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
            streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
            contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
            paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
            payment_method = st.selectbox("Payment Method", [
                "Electronic check", "Mailed check",
                "Bank transfer (automatic)", "Credit card (automatic)"
            ])
            monthly_charges = st.number_input("Monthly Charges ($)", 0.0, 150.0, 70.0)
            total_charges = st.number_input("Total Charges ($)", 0.0, 10000.0, monthly_charges * tenure)

        submit_button = st.form_submit_button("Predict Churn")

        if submit_button:
            # Prepare request
            features = {
                "gender": gender,
                "SeniorCitizen": senior_citizen,
                "Partner": partner,
                "Dependents": dependents,
                "tenure": tenure,
                "PhoneService": phone_service,
                "MultipleLines": multiple_lines,
                "InternetService": internet_service,
                "OnlineSecurity": online_security,
                "OnlineBackup": online_backup,
                "DeviceProtection": device_protection,
                "TechSupport": tech_support,
                "StreamingTV": streaming_tv,
                "StreamingMovies": streaming_movies,
                "Contract": contract,
                "PaperlessBilling": paperless_billing,
                "PaymentMethod": payment_method,
                "MonthlyCharges": monthly_charges,
                "TotalCharges": total_charges
            }

            try:
                response = requests.post(f"{api_url}/predict", json=features, timeout=10)

                if response.status_code == 200:
                    result = response.json()

                    st.success("Prediction Complete!")

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Churn Prediction", "Yes" if result["churn_prediction"] == 1 else "No")
                    with col2:
                        st.metric("Churn Probability", f"{result['churn_probability']:.2%}")
                    with col3:
                        risk_color = {"Low": "🟢", "Medium": "🟡", "High": "🔴"}
                        st.metric("Risk Level", f"{risk_color.get(result['risk_level'], '')} {result['risk_level']}")

                    # Visual gauge
                    fig = go.Figure(go.Indicator(
                        mode="gauge+number+delta",
                        value=result['churn_probability'] * 100,
                        title={'text': "Churn Probability (%)"},
                        domain={'x': [0, 1], 'y': [0, 1]},
                        gauge={
                            'axis': {'range': [0, 100]},
                            'bar': {'color': "darkblue"},
                            'steps': [
                                {'range': [0, 30], 'color': "#00cc96"},
                                {'range': [30, 60], 'color': "#ffa15a"},
                                {'range': [60, 100], 'color': "#ef553b"}
                            ],
                            'threshold': {
                                'line': {'color': "red", 'width': 4},
                                'thickness': 0.75,
                                'value': 80
                            }
                        }
                    ))
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.error(f"Prediction failed: {response.text}")
            except Exception as e:
                st.error(f"Error making prediction: {e}")

# Tab 3: Feature Importance
with tab3:
    st.header("Feature Importance Analysis")

    try:
        data_path = Path(__file__).parents[2] / config['paths']['data_path']
        if data_path.exists():
            df = load_data(str(data_path))

            # Correlation with churn
            st.subheader("Feature Correlation with Churn")
            numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
            correlations = df[numeric_cols].corr()['Churn'].drop('Churn').sort_values(ascending=False)

            fig = px.bar(
                x=correlations.values,
                y=correlations.index,
                orientation='h',
                title="Feature Correlation with Churn",
                labels={'x': 'Correlation', 'y': 'Feature'},
                color=correlations.values,
                color_continuous_scale='RdYlGn'
            )
            st.plotly_chart(fig, use_container_width=True)

            # Contract type analysis
            st.subheader("Churn Rate by Contract Type")
            contract_churn = df.groupby('Contract')['Churn'].mean().sort_values(ascending=False)
            fig = px.bar(
                x=contract_churn.index,
                y=contract_churn.values * 100,
                title="Churn Rate by Contract Type",
                labels={'x': 'Contract Type', 'y': 'Churn Rate (%)'},
                color=contract_churn.values,
                color_continuous_scale='Reds'
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Dataset not found")
    except Exception as e:
        st.error(f"Error analyzing features: {e}")

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("**MLOps Churn Prediction Engine v1.0.0**")
st.sidebar.markdown("Built with XGBoost, FastAPI, MLflow & Streamlit")
