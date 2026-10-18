import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_absolute_error, classification_report, roc_auc_score
from xgboost import XGBClassifier
from feature_engineering import build_feature_pipeline

def train_fraud_model():
    # 1. Load dataset
    data_path = "data/raw/creditcard.csv"
    if not os.path.exists(data_path):
        print(f"❌ Dataset not found at {data_path}. Please run generate_data.py first.")
        return

    print("Loading dataset...")
    df = pd.read_csv(data_path)

    # 2. Apply Feature Engineering Pipeline
    print("Applying feature engineering pipeline...")
    df = build_feature_pipeline(df)

    # 3. Define Features and Target
    # Include engineered features + principal components (V1-V28) if available
    features = ['Amount', 'Time', 'Log_Amount', 'Hour', 'Is_Night', 'Time_Delta', 'Tx_Count_1H', 'Amount_Sum_24H', 'Amount_Ratio_24H']
    v_features = [col for col in df.columns if col.startswith('V')]
    features.extend(v_features)

    # Ensure all selected features exist in the dataframe
    features = [f for f in features if f in df.columns]

    X = df[features]
    y = df['Class']

    # 4. Train/Test Split
    print("Splitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # 5. Initialize and Train XGBoost Classifier
    print("Training XGBoost Fraud Detection Model...")
    # Handle extreme class imbalance using scale_pos_weight
    fraud_ratio = (len(y_train) - sum(y_train)) / sum(y_train) if sum(y_train) > 0 else 1.0
    
    model = XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        scale_pos_weight=fraud_ratio,
        random_state=42,
        eval_metric='logloss'
    )
    
    model.fit(X_train, y_train)

    # 6. Evaluate Model
    print("\n--- Evaluating Model Performance ---")
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Metrics requested
    acc_score = accuracy_score(y_test, y_pred)
    mae_score = mean_absolute_error(y_test, y_pred)
    auc_score = roc_auc_score(y_test, y_prob)

    print(f"Accuracy Score:        {acc_score:.4f}")
    print(f"Mean Absolute Error:   {mae_score:.4f}")
    print(f"ROC-AUC Score:         {auc_score:.4f}")
    
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred))

    # 7. Save Model Artifact
    os.makedirs("models", exist_ok=True)
    model_path = "models/xgboost_fraud_model.pkl"
    joblib.dump(model, model_path)
    print(f"\n✅ Model successfully saved to '{model_path}'")

if __name__ == "__main__":
    train_fraud_model()