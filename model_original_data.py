import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import sys

# -----------------------------
# Feature Engineering Functions
# -----------------------------
def reduce_memory_usage(df):
    """Downcasts numeric columns to reduce memory usage."""
    for col in df.select_dtypes(include=['int64', 'float64']).columns:
        df[col] = pd.to_numeric(df[col], downcast='float')
    return df

def advanced_features(df, is_train=False, target_series=None):
    """
    Create advanced features for the weather dataset.
    Assumes input columns: id, day, pressure, maxtemp, temparature, mintemp,
    dewpoint, humidity, cloud, sunshine, winddirection, windspeed, rainfall.
    For training, 'rainfall' is the target and lag features are computed.
    Drops columns: 'id', 'day', 'date' (not needed for modeling).
    """
    df = df.copy()
    
    # a) Date features
    if "day" in df.columns:
        base_date = pd.to_datetime("2023-01-01")
        df["date"] = base_date + pd.to_timedelta(df["day"] - 1, unit="D")
        df["month"] = df["date"].dt.month
        df["day_of_year"] = df["date"].dt.dayofyear
        df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
        df["quarter"] = df["date"].dt.quarter
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
    
    # d) Lag features for rainfall (for training only)
    if is_train and "rainfall" in df.columns:
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
    
    # e) Drop columns that are not needed
    df.drop(["id", "day", "date"], axis=1, inplace=True, errors="ignore")
    
    return df

def advanced_features_test(df):
    """
    Create advanced features for the test dataset while preserving the 'id' column.
    Assumes input columns: id, day, pressure, maxtemp, temparature, mintemp,
    dewpoint, humidity, cloud, sunshine, winddirection, windspeed.
    No rainfall is available in test data.
    """
    df = df.copy()
    
    # Preserve id column
    ids = df["id"]
    
    # a) Date features
    if "day" in df.columns:
        base_date = pd.to_datetime("2023-01-01")
        df["date"] = base_date + pd.to_timedelta(df["day"] - 1, unit="D")
        df["month"] = df["date"].dt.month
        df["day_of_year"] = df["date"].dt.dayofyear
        df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
        df["quarter"] = df["date"].dt.quarter
        df["day_sin"] = np.sin(2 * np.pi * df["day_of_year"] / 365)
        df["day_cos"] = np.cos(2 * np.pi * df["day_of_year"] / 365)
        df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    
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
    if "pressure" in df.columns and "winddirection" in df.columns:
        df["pressure_wind_interaction"] = df["pressure"] * df["winddirection"]
    if "temparature" in df.columns and "pressure" in df.columns:
        df["temp_pressure_ratio"] = df["temparature"] / (df["pressure"] + 1e-3)
    if "windspeed" in df.columns and "pressure" in df.columns:
        df["wind_pressure_ratio"] = df["windspeed"] / (df["pressure"] + 1e-3)
    
    # d) Lag features: not available for test data; set to 0.
    df["gap_before_rain"] = 0
    df["gap_after_rain"] = 0
    
    # e) Drop unnecessary columns but preserve id.
    df.drop(["day", "date"], axis=1, inplace=True, errors="ignore")
    
    # f) Reorder columns (excluding id) to match the training FE data.
    desired_order = ['pressure', 'maxtemp', 'dewpoint', 'humidity', 'cloud', 'sunshine',
                     'winddirection', 'windspeed', 'month', 'day_sin', 'day_cos', 'temp_range',
                     'temp_dew_diff', 'humidity_cloud_ratio', 'sunshine_cloud_ratio',
                     'gap_before_rain', 'gap_after_rain']
    df = df[desired_order]
    
    # Insert id as the first column.
    df.insert(0, "id", ids)
    
    return df

# -----------------------------
# Training Phase
# -----------------------------
# 1) Load the original training dataset (non–feature engineered)
train_data = pd.read_csv("Data/train.csv")
train_data = reduce_memory_usage(train_data)
print("✅ Training data loaded. Shape:", train_data.shape)
print("Columns:", train_data.columns.tolist())

# 2) Apply feature engineering to the training data (drops id)
train_fe = advanced_features(train_data, is_train=True, target_series=train_data["rainfall"])
print("Enhanced training data shape:", train_fe.shape)
print(train_fe.head(5))

# 3) Define features and target from the FE training data
features = ['pressure', 'maxtemp', 'dewpoint', 'humidity', 'cloud', 'sunshine',
            'winddirection', 'windspeed', 'month', 'day_sin', 'day_cos', 'temp_range',
            'temp_dew_diff', 'humidity_cloud_ratio', 'sunshine_cloud_ratio',
            'gap_before_rain', 'gap_after_rain']

missing_feats = [f for f in features if f not in train_fe.columns]
if missing_feats:
    raise KeyError(f"The following features are missing from the FE training data: {missing_feats}")

X = train_fe[features].copy()
y = train_fe["rainfall"].copy()

print("\nFE Training Features shape:", X.shape)
print("FE Training Target shape:", y.shape)

# 4) Train-test split (stratified)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                     random_state=42, stratify=y)
print("\nTraining set shape:", X_train.shape)
print("Test set shape:", X_test.shape)

target_counts = y_train.value_counts()
print("\nTraining target distribution:")
print(target_counts)
if target_counts.shape[0] < 2:
    print("\n❌ Training data contains only one class. Logistic Regression requires at least 2 classes.")
    sys.exit(1)

# 5) Build pipeline with Logistic Regression and GridSearchCV
pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler()),
    ('clf', LogisticRegression(solver='liblinear', random_state=42))
])
param_grid = {
    'clf__C': [0.01, 0.1, 1, 10, 100],
    'clf__penalty': ['l1', 'l2']
}
grid = GridSearchCV(pipeline, param_grid, cv=5, scoring='roc_auc', n_jobs=-1, error_score='raise')
grid.fit(X_train, y_train)
print("\nBest ROC AUC Score from CV: {:.4f}".format(grid.best_score_))
print("Best Parameters:", grid.best_params_)

# 6) Evaluate on the held-out test set (from training split)
best_model = grid.best_estimator_
y_test_pred_proba = best_model.predict_proba(X_test)[:, 1]
y_test_pred = best_model.predict(X_test)
test_auc = roc_auc_score(y_test, y_test_pred_proba)
print("\nTest ROC AUC Score: {:.4f}".format(test_auc))
print("\nClassification Report on Test Set:")
print(classification_report(y_test, y_test_pred))
cv_scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring='roc_auc')
print("\nCross-validated ROC AUC scores on Training Set:", cv_scores)
print("Mean ROC AUC on Training Set:", np.mean(cv_scores))

# Plotting performance graphs
# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_test_pred_proba)
plt.figure()
plt.plot(fpr, tpr, label=f"ROC curve (AUC = {test_auc:.4f})")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.show()

# Confusion Matrix
cm = confusion_matrix(y_test, y_test_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=best_model.classes_)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()

# Box Plot for CV Scores
plt.figure()
plt.boxplot(cv_scores, patch_artist=True)
plt.title("Cross-Validated ROC AUC Scores")
plt.ylabel("ROC AUC Score")
plt.show()

# -----------------------------
# Test Prediction Phase
# -----------------------------
# Load the raw test file (non–feature engineered) that includes the "id" column.
test_raw = pd.read_csv("Data/test.csv")
test_raw = reduce_memory_usage(test_raw)
print("\n✅ Raw Test Data loaded. Shape:", test_raw.shape)
print("Columns:", test_raw.columns.tolist())

# Apply feature engineering to the test data while preserving "id"
test_fe = advanced_features_test(test_raw)
print("Enhanced Test Data shape:", test_fe.shape)
print(test_fe.head(5))

# Define features for prediction (exclude "id")
features = ['pressure', 'maxtemp', 'dewpoint', 'humidity', 'cloud', 'sunshine',
            'winddirection', 'windspeed', 'month', 'day_sin', 'day_cos', 'temp_range',
            'temp_dew_diff', 'humidity_cloud_ratio', 'sunshine_cloud_ratio',
            'gap_before_rain', 'gap_after_rain']
X_test_new = test_fe[features].copy()

# Predict probabilities on test data
test_pred_proba = best_model.predict_proba(X_test_new)[:, 1]

# Save predictions: two columns "id" (from test data) and "predicted_proba"
submission = pd.DataFrame({"id": test_fe["id"], "predicted_proba": test_pred_proba})
output_csv = "Data/test_predictions_original.csv"
submission.to_csv(output_csv, index=False)
print(f"\nTest predictions saved to {output_csv}")