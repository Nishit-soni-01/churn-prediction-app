import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go
import plotly.express as px



st.set_page_config(page_title="Churn Predictor Pro", page_icon="📉", layout="wide")

# --- 2. Load Model and Scaler ---
@st.cache_resource
def load_components():
    with open('svm_churn_model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    with open('scaler.pkl', 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)
    return model, scaler

model, scaler = load_components()

# --- 3. Sidebar Inputs ---
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3121/3121693.png", width=100) # Placeholder logo
st.sidebar.title("Customer Profile")
st.sidebar.markdown("Adjust the metrics below to simulate a customer's profile.")

# Group inputs logically
with st.sidebar.expander("Usage Metrics", expanded=True):
    tenure = st.slider("Tenure (Months)", min_value=0, max_value=72, value=24)
    support_calls = st.slider("Customer Support Calls", min_value=0, max_value=10, value=2)

with st.sidebar.expander("Financial Metrics", expanded=True):
    monthly_charges = st.slider("Monthly Charges ($)", min_value=15.0, max_value=120.0, value=75.0)
    total_charges = st.slider("Total Charges ($)", min_value=15.0, max_value=8000.0, value=1800.0)

# Create input dataframe
input_data = pd.DataFrame({
    'Tenure_Months': [tenure],
    'Monthly_Charges': [monthly_charges],
    'Total_Charges': [total_charges],
    'Support_Calls': [support_calls]
})

# --- 4. Robust Prediction Logic ---
scaled_input = scaler.transform(input_data)
prediction = model.predict(scaled_input)[0]

# Safely attempt to get probabilities so the app doesn't crash
try:
    probabilities = model.predict_proba(scaled_input)[0]
    churn_prob = probabilities[1] * 100
except AttributeError:
    # Fallback if the SVM model was trained without probability=True
    churn_prob = 100.0 if prediction == 1 else 0.0
    st.sidebar.warning("⚠️ Model lacks probability scoring. Gauge will only show 0 or 100.")

# --- 5. Main Dashboard UI ---
st.title("📉 Customer Churn Prediction Dashboard")
st.markdown("Leveraging **Support Vector Machine (SVM)** to identify at-risk customers in real-time.")
st.divider()

# Top Row: Prediction Result and Gauge Chart
col1, col2 = st.columns([1, 1.5])

with col1:
    st.markdown("### Risk Assessment")
    if prediction == 1:
        st.error("#### 🚨 High Churn Risk")
        st.write("This customer exhibits patterns strongly associated with churning. Immediate retention efforts are recommended.")
    else:
        st.success("#### ✅ Low Churn Risk")
        st.write("This customer is likely to remain. Standard engagement strategies are sufficient.")
        
    # Display quick metrics
    # Display quick metrics with both USD and INR
    ltv_usd = total_charges + (monthly_charges * 12)
    exchange_rate = 83.0 # You can update this to the exact current rate
    ltv_inr = ltv_usd * exchange_rate
    
    st.metric(label="Expected Lifetime Value (Est.)", 
              value=f"${ltv_usd:,.2f} | ₹{ltv_inr:,.2f}")

with col2:
    # Build an interactive Plotly Gauge Chart for Probability
    fig_gauge = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = churn_prob,
        number = {'suffix': "%", 'font': {'size': 40}},
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Calculated Risk Score", 'font': {'size': 20}},
        gauge = {
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "rgba(0,0,0,0.8)"}, # The needle
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 30], 'color': "#a8e6cf"},   # Soft Green
                {'range': [30, 70], 'color': "#ffd3b6"},   # Soft Orange/Yellow
                {'range': [70, 100], 'color': "#ff8b94"}], # Soft Red
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 50} 
        }
    ))
    
    # Adjusted margins to ensure it doesn't get clipped
    fig_gauge.update_layout(height=300, margin=dict(l=30, r=30, t=50, b=30))
    st.plotly_chart(fig_gauge, use_container_width=True)

st.divider()

# Bottom Row: Contextual Data Visualizations
st.markdown("### Customer Context")
col3, col4 = st.columns(2)

with col3:
    # A bar chart comparing this customer's monthly charge to hypothetical industry averages
    chart_data = pd.DataFrame({
        'Category': ['This Customer', 'Average Retained', 'Average Churned'],
        'Monthly Charge ($)': [monthly_charges, 55.0, 85.0]
    })
    fig_bar = px.bar(chart_data, x='Category', y='Monthly Charge ($)', 
                     title="Monthly Spend vs. Averages",
                     color='Category',
                     color_discrete_map={'This Customer': '#1f77b4', 'Average Retained': '#2ca02c', 'Average Churned': '#d62728'})
    st.plotly_chart(fig_bar, use_container_width=True)

with col4:
    # A scatter plot simulation to show where the customer sits
    np.random.seed(42)
    sim_tenure = np.random.normal(24, 15, 100).clip(0, 72)
    sim_charges = np.random.normal(65, 20, 100).clip(15, 120)
    sim_churn = np.where(sim_tenure < 12, 'High Risk', 'Low Risk')
    
    sim_df = pd.DataFrame({'Tenure': sim_tenure, 'Monthly Charges': sim_charges, 'Status': sim_churn})
    
    fig_scatter = px.scatter(sim_df, x='Tenure', y='Monthly Charges', color='Status',
                             title="Customer Positioning",
                             color_discrete_map={'High Risk': 'red', 'Low Risk': 'green'},
                             opacity=0.4)
    
    # Add the current user as a prominent marker
    fig_scatter.add_scatter(x=[tenure], y=[monthly_charges], mode='markers', 
                            marker=dict(size=15, color='black', symbol='star'),
                            name='Current Customer')
                            
    st.plotly_chart(fig_scatter, use_container_width=True)
