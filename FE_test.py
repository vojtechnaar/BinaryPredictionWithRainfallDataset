import pandas as pd
import numpy as np

def feature_engineering_test_data(df):
    """
    Transform raw test data to match the feature‐engineered training set,
    while preserving the 'id' column.
    Assumes input columns: id, day, pressure, maxtemp, temparature, mintemp,
    dewpoint, humidity, cloud, sunshine, winddirection, windspeed.
    """
    df = df.copy()
    
    # Preserve id column separately.
    ids = df["id"]
    
    # a) Date features
    if "day" in df.columns:
        base_date = pd.to_datetime("2023-01-01")
        df["date"] = base_date + pd.to_timedelta(df["day"] - 1, unit="D")
        df["month"] = df["date"].dt.month
        day_of_year = df["date"].dt.dayofyear
        df["day_sin"] = np.sin(2 * np.pi * day_of_year / 365)
        df["day_cos"] = np.cos(2 * np.pi * day_of_year / 365)
    
    # b) Temperature features
    if "maxtemp" in df.columns and "mintemp" in df.columns:
        df["temp_range"] = df["maxtemp"] - df["mintemp"]
    if "temparature" in df.columns and "dewpoint" in df.columns:
        df["temp_dew_diff"] = df["temparature"] - df["dewpoint"]
    
    # c) Interaction/ratio features
    if "humidity" in df.columns and "cloud" in df.columns:
        df["humidity_cloud_ratio"] = df["humidity"] / (df["cloud"] + 1e-3)
    if "sunshine" in df.columns and "cloud" in df.columns:
        df["sunshine_cloud_ratio"] = df["sunshine"] / (df["cloud"] + 1e-3)
    
    # d) Lag features (set to 0 for test data)
    df["gap_before_rain"] = 0
    df["gap_after_rain"] = 0
    
    # e) Drop unnecessary columns.
    # Drop columns that won't be used in predictions.
    df.drop(["day", "date", "temparature", "mintemp"], axis=1, inplace=True, errors="ignore")
    
    # f) Reorder columns to match the training feature-engineered data.
    # The desired order (without 'id') is:
    desired_order = ['pressure', 'maxtemp', 'dewpoint', 'humidity', 'cloud', 'sunshine',
                     'winddirection', 'windspeed', 'month', 'day_sin', 'day_cos', 'temp_range',
                     'temp_dew_diff', 'humidity_cloud_ratio', 'sunshine_cloud_ratio',
                     'gap_before_rain', 'gap_after_rain']
    df = df[desired_order]
    
    # g) Insert the preserved 'id' column at the beginning.
    df.insert(0, "id", ids)
    
    return df

# -----------------------------
# Putting It All Together for Test Data
# -----------------------------
# Load your raw test dataset.
# It should contain: id, day, pressure, maxtemp, temparature, mintemp, dewpoint, 
# humidity, cloud, sunshine, winddirection, windspeed
df_test = pd.read_csv("Data/test.csv")
print("✅ Raw test dataset loaded. Shape:", df_test.shape)
print("Columns:", df_test.columns.tolist())

# Apply feature engineering to the test data.
df_test_fe = feature_engineering_test_data(df_test)
print("Enhanced test dataset shape:", df_test_fe.shape)
print(df_test_fe.head(5))

# Save the enhanced test dataset (with 'id' preserved) to a CSV file.
output_csv = "Data/test_feature_engineered.csv"
df_test_fe.to_csv(output_csv, index=False)
print(f"Enhanced test dataset saved to {output_csv}")