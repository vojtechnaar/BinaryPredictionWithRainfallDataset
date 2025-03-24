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
# 1) Load the engineered training dataset
# -----------------------------
train_data = pd.read_csv("Data/train_feature_engineered.csv")
print("✅ Training data loaded. Shape:", train_data.shape)
print("Columns:", train_data.columns.tolist())

# -----------------------------
# 2) Define Features and Target from training data
# -----------------------------
features = ['pressure', 'maxtemp', 'dewpoint', 'humidity', 'cloud', 'sunshine',
            'winddirection', 'windspeed', 'month', 'day_sin', 'day_cos', 'temp_range',
            'temp_dew_diff', 'humidity_cloud_ratio', 'sunshine_cloud_ratio',
            'gap_before_rain', 'gap_after_rain']

missing_feats = [f for f in features if f not in train_data.columns]
if missing_feats:
    raise KeyError(f"The following features are missing from the training data: {missing_feats}")

X = train_data[features].copy()
y = train_data["rainfall"].copy()

print("\nTraining Features shape:", X.shape)
print("Training Target shape:", y.shape)

# -----------------------------
# 3) Create a Train–Validation Split for Evaluation
# -----------------------------
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, 
                                                  random_state=42, stratify=y)
print("\nTraining set shape:", X_train.shape)
print("Validation set shape:", X_val.shape)

target_counts = y_train.value_counts()
print("\nTraining target distribution:")
print(target_counts)
if target_counts.shape[0] < 2:
    print("\n❌ Training data contains only one class. Logistic Regression requires at least 2 classes.")
    sys.exit(1)

# -----------------------------
# 4) Build a Pipeline with Logistic Regression and Hyperparameter Tuning
# -----------------------------
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

# -----------------------------
# 5) Evaluate the Model on the Validation Set
# -----------------------------
best_model = grid.best_estimator_
y_val_pred_proba = best_model.predict_proba(X_val)[:, 1]
y_val_pred = best_model.predict(X_val)

val_auc = roc_auc_score(y_val, y_val_pred_proba)
print("\nValidation ROC AUC Score: {:.4f}".format(val_auc))
print("\nClassification Report on Validation Set:")
print(classification_report(y_val, y_val_pred))

cv_scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring='roc_auc')
print("\nCross-validated ROC AUC scores on Training Set:", cv_scores)
print("Mean ROC AUC on Training Set:", np.mean(cv_scores))

# -----------------------------
# 6) Plot Performance Graphs (Validation)
# -----------------------------
# ROC Curve
fpr, tpr, thresholds = roc_curve(y_val, y_val_pred_proba)
plt.figure()
plt.plot(fpr, tpr, label=f"ROC curve (AUC = {val_auc:.4f})")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Validation")
plt.legend(loc="lower right")
plt.show()

# Confusion Matrix
cm = confusion_matrix(y_val, y_val_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=best_model.classes_)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix - Validation")
plt.show()

# Box Plot for Cross-Validation Scores
plt.figure()
plt.boxplot(cv_scores, patch_artist=True)
plt.title("Cross-Validated ROC AUC Scores on Training Set")
plt.ylabel("ROC AUC Score")
plt.show()

# -----------------------------
# 7) Load Your Separate Test File for Predictions
# -----------------------------
# This test file should be the feature-engineered test dataset and include the 'id' column.
test_file_path = "Data/test_feature_engineered.csv"  
test_data = pd.read_csv(test_file_path)
print("\n✅ Separate Test Data loaded. Shape:", test_data.shape)
print("Columns:", test_data.columns.tolist())

# Check if required features exist in the test data
missing_feats_test = [f for f in features if f not in test_data.columns]
if missing_feats_test:
    raise KeyError(f"The following features are missing from the test data: {missing_feats_test}")

X_test_new = test_data[features].copy()

# -----------------------------
# 8) Predict Probabilities on the Test Data
# -----------------------------
test_pred_proba = best_model.predict_proba(X_test_new)[:, 1]

# -----------------------------
# 9) Save the Predictions to a CSV File with 2 columns: 'id' and 'predicted_proba'
# -----------------------------
submission = pd.DataFrame({"id": test_data["id"], "predicted_proba": test_pred_proba})
output_csv = "Data/test_predictions.csv"
submission.to_csv(output_csv, index=False)
print(f"\nSeparate test predictions saved to {output_csv}")