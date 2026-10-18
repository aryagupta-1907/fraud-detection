import os
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import shap
import matplotlib.pyplot as plt

def evaluate_model():
    # Paths
    model_path = "models/xgboost_fraud_model.pkl"
    test_data_path = "data/raw/creditcard.csv" # Or your test split path
    
    if not os.path.exists(model_path):
        print("❌ Model not found! Please run train.py first.")
        return

    if not os.path.exists(test_data_path):
        print("❌ Test dataset not found! Please check your data directory.")
        return

    # Load model and data
    print("Loading model and data...")
    model = joblib.load(model_path)
    df = pd.read_csv(test_data_path)

    # Basic preprocessing to match training features
    if 'Log_Amount' not in df.columns:
        df['Log_Amount'] = np.log1p(df['Amount'])
    if 'Hour' not in df.columns:
        df['Hour'] = (df['Time'] // 3600) % 24
    if 'Is_Night' not in df.columns:
        df['Is_Night'] = df['Hour'].apply(lambda x: 1 if (x < 6 or x > 22) else 0)

    # Define features and target (matching train.py)
    features = ['Amount', 'Time', 'Log_Amount', 'Hour', 'Is_Night']
    # Include V1-V28 if they exist in the dataset
    v_features = [col for col in df.columns if col.startswith('V')]
    features.extend(v_features)

    X = df[features]
    y = df['Class']

    # Predictions
    print("Evaluating model performance...")
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]

    # Metrics
    print("\n--- Classification Report ---")
    print(classification_report(y, y_pred))
    
    auc_score = roc_auc_score(y, y_prob)
    print(f"ROC-AUC Score: {auc_score:.4f}")

    # --- SHAP Global Evaluation ---
    print("\nGenerating SHAP Global Feature Importance Summary...")
    explainer = shap.TreeExplainer(model)
    # Using a sample of the data for faster SHAP computation if dataset is large
    X_sample = X.sample(n=min(1000, len(X)), random_state=42)
    shap_values = explainer(X_sample)

    # Create and save SHAP summary plot
    os.makedirs("models", exist_ok=True)
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_sample, show=False)
    plt.title("SHAP Global Feature Importance", fontsize=14)
    plt.tight_layout()
    plt.savefig("models/shap_summary_plot.png")
    plt.close()
    print("✅ SHAP summary plot saved successfully as 'models/shap_summary_plot.png'.")

if __name__ == "__main__":
    evaluate_model()