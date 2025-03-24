import pandas as pd
import numpy as np

# -----------------------------
# 1) Function to reduce memory usage
# -----------------------------
def reduce_memory_usage(df):
    """Downcasts numeric columns to reduce memory usage."""
    for col in df.select_dtypes(include=['int64', 'float64']).columns:
        df[col] = pd.to_numeric(df[col], downcast='float')
    return df

# -----------------------------
# 2) Advanced Feature Engineering Function
# -----------------------------
def advanced_features(df, is_train=False, target_series=None):
    """
    Create advanced features for the weather dataset.
    Assumes columns: id, day, pressure, maxtemp, temparature, mintemp,
    dewpoint, humidity, cloud, sunshine, winddirection, windspeed, rainfall.
    The target "rainfall" is left unchanged.
    """
    df = df.copy()
    
    # a) Date features: Convert 'day' to a proper date.
    if "day" in df.columns:
        base_date = pd.to_datetime("2023-01-01")
        df["date"] = base_date + pd.to_timedelta(df["day"] - 1, unit="D")
        df["month"] = df["date"].dt.month
        df["day_of_year"] = df["date"].dt.dayofyear
        df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
        df["quarter"] = df["date"].dt.quarter
        
        # Periodic features (using sine/cosine)
        df["day_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365)
        df["day_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 365)
        df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    
    # b) Temperature features
    if "maxtemp" in df.columns and "mintemp" in df.columns:
        df["temp_range"] = df["maxtemp"] - df["mintemp"]
    if all(col in df.columns for col in ["maxtemp", "temparature", "mintemp"]):
        df["avg_temp"] = df[["maxtemp", "temparature", "mintemp"]].mean(axis=1)
    if "temparature" in df.columns and "dewpoint" in df.columns:
        df["temp_dew_diff"] = df["temparature"] - df["dewpoint"]
    
    # c) Interaction/ratio features
    if "humidity" in df.columns and "cloud" in df.columns:
        df["humidity_cloud_ratio"] = df["humidity"] / (df["cloud"] + 1e-3)
    if "sunshine" in df.columns and "cloud" in df.columns:
        df["sunshine_cloud_ratio"] = df["sunshine"] / (df["cloud"] + 1e-3)
    if "pressure" in df.columns and "winddirection" in df.columns:
        df["pressure_wind_interaction"] = df["pressure"] * df["winddirection"]
    if "temparature" in df.columns and "pressure" in df.columns:
        df["temp_pressure_ratio"] = df["temparature"] / (df["pressure"] + 1e-3)
    if "windspeed" in df.columns and "pressure" in df.columns:
        df["wind_pressure_ratio"] = df["windspeed"] / (df["pressure"] + 1e-3)
    
    # d) Lag features for rainfall (only for training)
    if is_train and "rainfall" in df.columns:
        # Optionally, use target_series if provided (should match df["rainfall"])
        if target_series is not None:
            df["rainfall"] = target_series.values
        df = df.sort_values("date").reset_index(drop=True)
        df["rain_prev"] = df["rainfall"].shift(1).fillna(0)
        df["rain_next"] = df["rainfall"].shift(-1).fillna(0)
        df["gap_before_rain"] = df.groupby((df["rain_prev"] != df["rainfall"]).cumsum()).cumcount()
        df["gap_after_rain"] = df[::-1].groupby((df["rain_next"] != df["rainfall"]).cumsum()).cumcount()[::-1]
        df.drop(["rain_prev", "rain_next"], axis=1, inplace=True)
    else:
        df["gap_before_rain"] = 0
        df["gap_after_rain"] = 0
    
    # e) Drop columns that are no longer needed (e.g., id, day, date)
    df.drop(["id", "day", "date"], axis=1, inplace=True, errors='ignore')
    
    return df

# -----------------------------
# 3) Remove Highly Correlated Features
# -----------------------------
def remove_highly_correlated(df, threshold=0.95):
    """
    Remove features that have a correlation coefficient greater than the threshold.
    """
    corr_matrix = df.corr().abs()
    # Only consider the upper triangle
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    print("\nDropping highly correlated features:", to_drop)
    return df.drop(columns=to_drop)

# -----------------------------
# 4) Outlier Treatment using IQR Method (skipping 'rainfall')
# -----------------------------
def treat_outliers_iqr(df):
    df = df.copy()
    for col in df.select_dtypes(include=[np.number]).columns:
        if col == 'rainfall':
            continue  # Skip binary target column
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df[col] = np.clip(df[col], lower, upper)
    return df

# -----------------------------
# 5) Putting It All Together
# -----------------------------
# Load the dataset, apply transformations, and save the enhanced dataset.
df = pd.read_csv("Data/train.csv")
df = reduce_memory_usage(df)
print("✅ Dataset loaded. Shape:", df.shape)

df_fe = advanced_features(df, is_train=True, target_series=df["rainfall"])
print("Enhanced dataset shape:", df_fe.shape)
print(df_fe.head(5))

df_fe_clean = remove_highly_correlated(df_fe, threshold=0.95)
print("After correlation removal, shape:", df_fe_clean.shape)

df_fe_clean = treat_outliers_iqr(df_fe_clean)
print("\nData after outlier treatment:")
print(df_fe_clean.head(10))

output_csv = "Data/train_feature_engineered.csv"
df_fe_clean.to_csv(output_csv, index=False)
print(f"\nEnhanced dataset saved to {output_csv}")