import pandas as pd

# Load from local file
print("Loading dataset...")
df = pd.read_csv("data/diabetic_data.csv")

# Show basic info
print("Dataset shape:", df.shape)
print("\nColumn names:")
print(df.columns.tolist())
print("\nFirst 5 rows:")
print(df.head())
print("\nTarget value counts:")
print(df['readmitted'].value_counts())

# Check for missing values
print("\nMissing values per column:")
print(df.isnull().sum()[df.isnull().sum() > 0])

# Check data types
print("\nData types:")
print(df.dtypes)

# Basic stats
print("\nBasic stats:")
print(df.describe())