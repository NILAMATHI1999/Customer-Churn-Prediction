import pandas as pd

# Load the dataset
file_path = "customer_churn.csv"
df = pd.read_csv(file_path)

# Show basic info
print("✅ Dataset Loaded")
print("\n📊 First 5 rows:")
print(df.head())

print("\n🧾 Dataset shape (rows, columns):")
print(df.shape)

print("\n🔍 Missing values in each column:")
print(df.isnull().sum())
