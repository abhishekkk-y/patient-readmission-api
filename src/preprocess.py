# Replace all ? with NaN
# Drop ID columns and columns with too many missing values
# Convert readmitted into a binary 0/1 target
# Convert age ranges to numbers
# Encode medication columns as 0/1
# Encode remaining text columns
# Split into train / validation / test sets

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os

def load_raw_data(filepath="data/diabetic_data.csv"):
    df = pd.read_csv(filepath)
    return df


def preprocess(df):

    # --- Step 1: Replace ? with NaN ---
    # The dataset uses ? instead of leaving cells blank
    df = df.replace('?', np.nan)

    # --- Step 2: Drop columns we don't need ---
    cols_to_drop = [
        'encounter_id',      # just an ID number, not a feature
        'patient_nbr',       # just a patient ID number
        'weight',            # over 96% missing
        'payer_code',        # insurance code, not clinically relevant here
        'medical_specialty', # too many missing values
        'diag_1',            # ICD codes with 700+ unique values, too complex for now
        'diag_2',
        'diag_3',
    ]
    df = df.drop(columns=cols_to_drop)

    # --- Step 3: Create binary target column ---
    # 1 = readmitted within 30 days (this is what we want to predict)
    # 0 = not readmitted within 30 days
    df['readmitted_30'] = (df['readmitted'] == '<30').astype(int)
    df = df.drop(columns=['readmitted'])

    # --- Step 4: Handle glucose and A1C test columns ---
    # These are 95% and 83% missing respectively
    # Instead of dropping them entirely, we capture whether the test
    # was done at all — that itself can be a useful signal
    df['glucose_tested'] = df['max_glu_serum'].notna().astype(int)
    df['a1c_tested'] = df['A1Cresult'].notna().astype(int)
    df = df.drop(columns=['max_glu_serum', 'A1Cresult'])

    # --- Step 5: Convert age ranges to numeric midpoints ---
    # [30-40) becomes 35, [50-60) becomes 55, etc.
    age_map = {
        '[0-10)':   5,
        '[10-20)': 15,
        '[20-30)': 25,
        '[30-40)': 35,
        '[40-50)': 45,
        '[50-60)': 55,
        '[60-70)': 65,
        '[70-80)': 75,
        '[80-90)': 85,
        '[90-100)': 95
    }
    df['age'] = df['age'].map(age_map)

    # --- Step 6: Encode gender as binary ---
    df['gender'] = df['gender'].map({'Male': 1, 'Female': 0})
    df['gender'] = df['gender'].fillna(0)

    # --- Step 7: Encode medication columns as binary ---
    # No = 0 (not on this medication)
    # Steady / Up / Down = 1 (on this medication, regardless of dose change)
    med_cols = [
        'metformin', 'repaglinide', 'nateglinide', 'chlorpropamide',
        'glimepiride', 'acetohexamide', 'glipizide', 'glyburide',
        'tolbutamide', 'pioglitazone', 'rosiglitazone', 'acarbose',
        'miglitol', 'troglitazone', 'tolazamide', 'examide',
        'citoglipton', 'insulin', 'glyburide-metformin',
        'glipizide-metformin', 'glimepiride-pioglitazone',
        'metformin-rosiglitazone', 'metformin-pioglitazone'
    ]
    for col in med_cols:
        df[col] = df[col].apply(lambda x: 0 if x == 'No' else 1)

    # --- Step 8: Encode change and diabetesMed ---
    df['change'] = df['change'].map({'No': 0, 'Ch': 1})
    df['diabetesMed'] = df['diabetesMed'].map({'No': 0, 'Yes': 1})

    # --- Step 9: Encode race with one-hot encoding ---
    # One-hot encoding turns one column with 5 values
    # into 5 separate 0/1 columns — one per race
    df['race'] = df['race'].fillna('Unknown')
    df = pd.get_dummies(df, columns=['race'], dtype=int)

    # --- Step 10: Drop any remaining rows with missing values ---
    df = df.dropna()

    return df


def split_data(df):
    X = df.drop(columns=['readmitted_30'])
    y = df['readmitted_30']

    # Split into 70% train, 15% validation, 15% test
    # stratify=y means each split has the same ratio of 0s and 1s
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


if __name__ == "__main__":

    print("Loading raw data...")
    df = load_raw_data()

    print("Preprocessing...")
    df_clean = preprocess(df)

    print("\nClean dataset shape:", df_clean.shape)
    print("\nTarget distribution after preprocessing:")
    print(df_clean['readmitted_30'].value_counts())
    print("\nColumns in clean dataset:")
    print(df_clean.columns.tolist())

    print("\nSplitting into train / val / test...")
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df_clean)

    print(f"Train size:      {X_train.shape}")
    print(f"Validation size: {X_val.shape}")
    print(f"Test size:       {X_test.shape}")

    # Save all splits to data/ folder
    X_train.to_csv("data/X_train.csv", index=False)
    X_val.to_csv("data/X_val.csv",   index=False)
    X_test.to_csv("data/X_test.csv", index=False)
    y_train.to_csv("data/y_train.csv", index=False)
    y_val.to_csv("data/y_val.csv",   index=False)
    y_test.to_csv("data/y_test.csv", index=False)

    print("\nAll splits saved to data/ folder.")