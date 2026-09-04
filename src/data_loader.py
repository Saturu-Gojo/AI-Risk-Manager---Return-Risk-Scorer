import os
import pandas as pd
from sklearn.model_selection import train_test_split

def load_raw_data(data_dir="."):
    """Loads raw train.csv and test.csv from data_dir."""
    train_path = os.path.join(data_dir, "train.csv")
    test_path = os.path.join(data_dir, "test.csv")
    
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"train.csv not found at {train_path}")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"test.csv not found at {test_path}")
        
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df

def split_train_val_test(train_df, target_col="returned", test_size=0.15, val_size=0.15):
    """
    Splits train_df into 70% Train, 15% Validation, and 15% Internal Test set chronologically (temporal split) based on order_id.
    """
    # Sort chronologically by order_id
    sorted_df = train_df.sort_values("order_id").reset_index(drop=True)
    n = len(sorted_df)
    
    train_end = int(n * (1.0 - val_size - test_size))
    val_end = int(n * (1.0 - test_size))
    
    train = sorted_df.iloc[:train_end].reset_index(drop=True)
    val = sorted_df.iloc[train_end:val_end].reset_index(drop=True)
    internal_test = sorted_df.iloc[val_end:].reset_index(drop=True)
    
    return train, val, internal_test

if __name__ == "__main__":
    train_df, test_df = load_raw_data()
    train, val, test = split_train_val_test(train_df)
    print(f"Raw Train: {train_df.shape}, Raw Test: {test_df.shape}")
    print(f"Split Train: {train.shape}, Split Val: {val.shape}, Split Internal Test: {test.shape}")
