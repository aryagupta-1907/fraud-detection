import os
import pandas as pd
import numpy as np

def generate_sample_raw_data(output_path='data/raw/creditcard.csv'):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    np.random.seed(42)
    n_samples = 3000
    
    time_col = np.sort(np.random.randint(0, 172800, n_samples))
    pca_columns = {f'V{i}': np.random.normal(0, 1, n_samples) for i in range(1, 29)}
    amount_col = np.random.exponential(75.0, n_samples)
    class_col = np.random.choice([0, 1], size=n_samples, p=[0.98, 0.02])
    
    data = {
        'Time': time_col,
        **pca_columns,
        'Amount': amount_col,
        'Class': class_col
    }
    
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
    print(f"Dataset generated at: {output_path} (Shape: {df.shape})")

if __name__ == "__main__":
    generate_sample_raw_data()