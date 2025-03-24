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
# 1) Load the engineered dataset
# -----------------------------
data = pd.read_csv("Data/train_feature_engineered.csv")
print("✅ Data loaded. Shape:", data.shape)
print("Columns:", data.columns.tolist())

# -----------------------------
# 2) Define Features and Target
# -----------------------------
features = ['pressure', 'maxtemp', 'dewpoint', 'humidity', 'cloud', 'sunshine',
            'winddirection', 'windspeed', 'month', 'day_sin', 'day_cos', 'temp_range',
            'temp_dew_diff', 'humidity_cloud_ratio', 'sunshine_cloud_ratio',
            'gap_before_rain', 'gap_after_rain']

# Check if all features exist:
missing_feats = [f for f in features if f not in data.columns]
if missing_feats:
    raise KeyError(f"The following features are missing from the data: {missing_feats}")

X = data[features].copy()
y = data["rainfall"].copy()

print("\nFeatures shape:", X.shape)
print("Target shape:", y.shape)

# -----------------------------
# 3) Train-Test Split (stratified)
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                     random_state=42, stratify=y)
print("\nTraining set shape:", X_train.shape)
print("Test set shape:", X_test.shape)

# -----------------------------
# 4) Check target distribution in training set
# -----------------------------
target_counts = y_train.value_counts()
print("\nTraining target distribution:")
print(target_counts)

if target_counts.shape[0] < 2:
    print("\n❌ Training data contains only one class. Logistic Regression requires at least 2 classes.")
    sys.exit(1)

# -----------------------------
# 5) Build a Pipeline with Logistic Regression and Hyperparameter Tuning
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
# 6) Evaluate on the Test Set
# -----------------------------
best_model = grid.best_estimator_
y_test_pred_proba = best_model.predict_proba(X_test)[:, 1]
y_test_pred = best_model.predict(X_test)

test_auc = roc_auc_score(y_test, y_test_pred_proba)
print("\nTest ROC AUC Score: {:.4f}".format(test_auc))
print("\nClassification Report on Test Set:")
print(classification_report(y_test, y_test_pred))

# -----------------------------
# 7) Optional: Cross-Validation Scores on Training Set
# -----------------------------
cv_scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring='roc_auc')
print("\nCross-validated ROC AUC scores on Training Set:", cv_scores)
print("Mean ROC AUC on Training Set:", np.mean(cv_scores))

# -----------------------------
# 8) Save Test Predictions to CSV
# -----------------------------
if "id" in X_test.columns:
    submission = pd.DataFrame({"id": X_test["id"], "rainfall": y_test_pred})
else:
    submission = pd.DataFrame({"id": np.arange(len(y_test_pred)), "rainfall": y_test_pred})

output_csv = "test_predictions.csv"
submission.to_csv(output_csv, index=False)
print(f"\nTest predictions saved to {output_csv}")

# -----------------------------
# 9) Plotting Model Performance Graphs
# -----------------------------
# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_test_pred_proba)
plt.figure()
plt.plot(fpr, tpr, label=f"ROC curve (AUC = {test_auc:.4f})")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic (ROC) Curve")
plt.legend(loc="lower right")
plt.show()

# Confusion Matrix
cm = confusion_matrix(y_test, y_test_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=best_model.classes_)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()

# Box Plot of Cross-Validation Scores
plt.figure()
plt.boxplot(cv_scores, patch_artist=True)
plt.title("Cross-Validated ROC AUC Scores")
plt.ylabel("ROC AUC Score")
plt.show()