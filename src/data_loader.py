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

def split_train_val_test(train_df, target_col="returned", test_size=0.15, val_size=0.15, random_state=42):
    """
    Splits train_df into 70% Train, 15% Validation, and 15% Internal Test set using stratified splitting.
    """
    train_val, internal_test = train_test_split(
        train_df, 
        test_size=test_size, 
        stratify=train_df[target_col], 
        random_state=random_state
    )
    
    val_relative_size = val_size / (1.0 - test_size)
    
    train, val = train_test_split(
        train_val,
        test_size=val_relative_size,
        stratify=train_val[target_col],
        random_state=random_state
    )
    
    return train.reset_index(drop=True), val.reset_index(drop=True), internal_test.reset_index(drop=True)

if __name__ == "__main__":
    train_df, test_df = load_raw_data()
    train, val, test = split_train_val_test(train_df)
    print(f"Raw Train: {train_df.shape}, Raw Test: {test_df.shape}")
    print(f"Split Train: {train.shape}, Split Val: {val.shape}, Split Internal Test: {test.shape}")
