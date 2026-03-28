# 📊 Customer Churn Prediction Pro

An interactive machine learning dashboard built with **Streamlit** to identify customers at risk of leaving. This project demonstrates a full data science pipeline: from synthetic data generation and model training to cloud deployment.

## 🚀 Live Demo
[View the Live App Here](https://churn-prediction-app-9xlvywg4ubj6aywmt8ztuq.streamlit.app/)

## 🛠️ Features
* **Real-time Inference:** Adjust customer metrics via sidebar sliders to see instant churn probability.
* **Risk Visualization:** Interactive Plotly gauge charts and scatter plots for market positioning.
* **Explainable AI (XAI):** A dedicated "Model Insights" section showing feature importance scores.
* **Dual-Currency Metrics:** Calculates Expected Lifetime Value (LTV) in both USD ($) and INR (₹).

## 🧠 Model & Dataset Information
* **Dataset:** The model is trained on a **custom synthetic dataset** of 2,000 samples. 
* **Logic:** The data simulates realistic churn behavior where high support calls, high monthly charges, and low tenure correlate with a higher risk of leaving.
* **Algorithm:** Support Vector Machine (SVM) with an RBF kernel and probability enabled.
* **Accuracy:** The model achieved a **94% accuracy** on the test set.

## 📦 Project Structure
* `app.py`: The main Streamlit application code.
* `model_training.ipynb`: The Jupyter notebook used for data generation, scaling, and training.
* `svm_churn_model.pkl`: The saved SVM model.
* `scaler.pkl`: The fitted StandardScaler to ensure input consistency.
* `requirements.txt`: Python dependencies (Streamlit, Scikit-learn, Pandas, Plotly).

## 💻 How to Run Locally
1. **Clone the repository:**
   ```bash
   git clone [https://github.com/Nishit-soni-01/churn-prediction-app.git](https://github.com/Nishit-soni-01/churn-prediction-app.git)
   cd churn-prediction-app
