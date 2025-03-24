import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math

# ------------------------
# 1) Load the dataset
# ------------------------
df = pd.read_csv("Data/train.csv")

print("✅ Data loaded.")
print("Shape:", df.shape)
print("Columns:", df.columns.tolist())

# ------------------------
# 2) Basic overview
# ------------------------
print("\n--- DataFrame Info ---")
df.info()

print("\n--- First 5 rows ---")
print(df.head())

# ------------------------
# 3) Missing values table
# ------------------------
print("\n--- Missing Values Table ---")
missing_counts = df.isna().sum()
print(missing_counts)

# ------------------------
# 4) Descriptive statistics for numeric columns
# ------------------------
print("\n--- Descriptive Statistics (Numeric) ---")
print(df.describe())

# ------------------------
# 5) Distribution plots for numeric columns (all in one figure)
# ------------------------
numeric_cols = df.select_dtypes(include=[np.number]).columns
n = len(numeric_cols)
ncols = math.ceil(math.sqrt(n))
nrows = math.ceil(n / ncols)

plt.figure(figsize=(ncols * 5, nrows * 4))
for i, col in enumerate(numeric_cols):
    plt.subplot(nrows, ncols, i + 1)
    sns.histplot(df[col].dropna(), kde=True)
    plt.title(f"Distribution of {col}")
    plt.xlabel(col)
    plt.ylabel("Count")
plt.tight_layout()
plt.show()

# ------------------------
# 6) Correlation Matrix: Heatmap and Table
# ------------------------
plt.figure(figsize=(8, 6))
corr = df[numeric_cols].corr()
sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Matrix")
plt.show()

print("\n--- Correlation Matrix Table ---")
print(corr)

# ------------------------
# 7) Scatter Plots: Relationship with 'rainfall'
# ------------------------
# Assume that 'rainfall' is the target variable you want to predict.
if 'rainfall' in numeric_cols:
    target = 'rainfall'
    other_vars = [col for col in numeric_cols if col != target]
    n_other = len(other_vars)
    ncols_scatter = math.ceil(math.sqrt(n_other))
    nrows_scatter = math.ceil(n_other / ncols_scatter)
    
    plt.figure(figsize=(ncols_scatter * 5, nrows_scatter * 4))
    for i, col in enumerate(other_vars):
        plt.subplot(nrows_scatter, ncols_scatter, i + 1)
        sns.scatterplot(x=df[col], y=df[target])
        plt.title(f"{target} vs {col}")
        plt.xlabel(col)
        plt.ylabel(target)
    plt.tight_layout()
    plt.show()
else:
    print("\nThe column 'rainfall' is not present in the numeric columns.")
    
print("\n✔️ EDA process finished.")