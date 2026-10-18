import pandas as pd
import numpy as np

def compute_log_amount(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes log1p of the Amount column to handle heavy right-skewed distributions.
    """
    df = df.copy()
    if 'Amount' in df.columns:
        df['Log_Amount'] = np.log1p(df['Amount'])
    return df

def compute_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extracts time of day (Hour) and categorical flags (Is_Night).
    Assumes 'Time' is elapsed seconds from start.
    """
    df = df.copy()
    if 'Time' in df.columns:
        df['Hour'] = (df['Time'] // 3600) % 24
        df['Is_Night'] = df['Hour'].apply(lambda h: 1 if (h < 6 or h > 22) else 0)
    return df

def compute_velocity_features(df: pd.DataFrame, user_col: str = 'User_ID') -> pd.DataFrame:
    """
    Computes dynamic sliding-window velocity metrics per user/cardholder:
    - Time_Delta: Seconds elapsed since the user's previous transaction.
    - Tx_Count_1H: Number of transactions by this user in the last 1 hour (3600 seconds).
    - Amount_Sum_24H: Total dollar amount spent by this user in the last 24 hours (86400 seconds).
    - Amount_Ratio_24H: Ratio of current transaction amount vs. average 24-hour spend.
    """
    df = df.copy()

    # If dataset lacks user tracking, create a dummy default user ID
    if user_col not in df.columns:
        df[user_col] = 'user_default'

    # Ensure dataset is chronologically sorted
    if 'Time' in df.columns:
        df = df.sort_values(by=['Time']).reset_index(drop=True)

    # Initialize new dynamic velocity feature columns
    df['Time_Delta'] = 0.0
    df['Tx_Count_1H'] = 1
    df['Amount_Sum_24H'] = df['Amount'] if 'Amount' in df.columns else 0.0
    df['Amount_Ratio_24H'] = 1.0

    # Calculate sliding window metrics per user
    for user_id, group in df.groupby(user_col):
        indices = group.index
        times = group['Time'].values
        amounts = group['Amount'].values if 'Amount' in group.columns else np.zeros(len(group))

        time_deltas = np.zeros(len(group))
        tx_1h = np.ones(len(group), dtype=int)
        amt_24h = amounts.copy()
        amt_ratios = np.ones(len(group))

        for i in range(len(group)):
            curr_time = times[i]
            curr_amt = amounts[i]

            # 1. Time since last transaction
            if i > 0:
                time_deltas[i] = curr_time - times[i - 1]

            # 2. Count of transactions in the last 1 hour (3600s)
            window_1h_mask = (times[:i+1] >= (curr_time - 3600))
            tx_1h[i] = np.sum(window_1h_mask)

            # 3. Sum & average of transaction amounts in last 24 hours (86400s)
            window_24h_mask = (times[:i+1] >= (curr_time - 86400))
            amt_24h[i] = np.sum(amounts[:i+1][window_24h_mask])
            avg_24h = np.mean(amounts[:i+1][window_24h_mask])

            # 4. Ratio of current amount vs. 24h average spend
            if avg_24h > 0:
                amt_ratios[i] = curr_amt / avg_24h

        # Assign computed values back to DataFrame
        df.loc[indices, 'Time_Delta'] = time_deltas
        df.loc[indices, 'Tx_Count_1H'] = tx_1h
        df.loc[indices, 'Amount_Sum_24H'] = amt_24h
        df.loc[indices, 'Amount_Ratio_24H'] = amt_ratios

    return df

def build_feature_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """
    Executes the full feature engineering pipeline sequentially.
    """
    df = compute_log_amount(df)
    df = compute_time_features(df)
    df = compute_velocity_features(df)
    return df

if __name__ == "__main__":
    # Test script with dummy mock transactions
    sample_data = pd.DataFrame({
        'User_ID': ['U101', 'U101', 'U101', 'U102'],
        'Time': [0, 1800, 3500, 100],  # Seconds
        'Amount': [25.0, 500.0, 1200.0, 45.0]
    })
    
    processed_df = build_feature_pipeline(sample_data)
    print("--- Processed Feature Pipeline Sample ---")
    print(processed_df[['User_ID', 'Time', 'Amount', 'Log_Amount', 'Time_Delta', 'Tx_Count_1H', 'Amount_Sum_24H', 'Amount_Ratio_24H']])