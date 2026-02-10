# -*- coding: utf-8 -*-
"""
Created on Mon Nov 10 10:37:53 2025
Group Project - Group#2
Project: Bicycle Theft Prediction

This script performs:
1. Data exploration
2. Statistical analysis
3. Visualization
4. Missing value handling
5. Categorical encoding
6. Train/Test split and scaling
7. Predictive modeling (Logistic Regression + Random Forest)
"""

# ============================ IMPORT LIBRARIES ============================
import pandas as pd                      # Handles tabular data
import numpy as np                       # Numerical operations
import matplotlib.pyplot as plt          # Charts and graphs
import seaborn as sns                    # Nice visualizations

from sklearn.model_selection import train_test_split     # Splits data
from sklearn.preprocessing import StandardScaler          # Normalization
from sklearn.linear_model import LogisticRegression        # Model 1
from sklearn.ensemble import RandomForestClassifier       # Model 2
from sklearn.metrics import (
    accuracy_score, confusion_matrix,
    classification_report, roc_curve, roc_auc_score, brier_score_loss
)
import pickle                     # <-- pickle for serialization (as required)
import warnings
from pathlib import Path
import json
from datetime import datetime, timezone

warnings.filterwarnings('ignore')  # Hides warnings for clean output
# ==============================================================================
# PART 1: DATA EXPLORATION
# ==============================================================================
print("===== PART 1: DATA EXPLORATION =====\n")

# -----------------------------------------------------------
# Requirement 1a: Load and inspect the dataset
# -----------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
file_path = BASE_DIR / "data" / "bicycle_thefts_open_data.csv"

data = pd.read_csv(file_path)
print("Requirement 1a: Dataset loaded successfully.")
print("----------------------------------------\n")


print("Requirement 1a: Data Elements Description")
print("----------------------------------------")
# Dataset structure + data types
print("Dataset Overview (Types & Non-Null Counts):")
print(data.info())


print("\nFirst 5 Rows:")
print(data.head())


# Loop through all columns to describe them clearly
print("\nColumn Descriptions and Value Ranges:")
for column in data.columns:
    print(f"\n- Column: '{column}' | Type: {data[column].dtype}")

    if data[column].dtype in ['int64', 'float64']:
        print(f"  Range: {data[column].min()} to {data[column].max()}")

    elif data[column].dtype == 'object':
        print(f"  Unique Values: {data[column].nunique()}")
        print(f"  Top 5 Values:\n{data[column].value_counts().head().to_string()}")
print("\n----------------------------------------\n")


# -----------------------------------------------------------
# Requirement 1b: Statistical assessments
# -----------------------------------------------------------
print("Requirement 1b: Statistical Assessments")
print("----------------------------------------")

numeric_data = data.select_dtypes(include=np.number)

print("\n===== Statistical Summary for Numeric Columns =====")
print("Mean values:")
print(numeric_data.mean())

print("\nMedian values:")
print(numeric_data.median())

print("\nStandard deviations:")
print(numeric_data.std())
print("---------------------------------------------------\n")


# Correlation matrix + heatmap
correlation_matrix = numeric_data.corr()

plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix of Numeric Features')
plt.show()

print("\nCorrelation matrix calculated and visualized.")
print("----------------------------------------\n")


# -----------------------------------------------------------
# Requirement 1c: Missing data evaluation
# -----------------------------------------------------------
# This section checks how many missing (NaN) values are in each column.
# Missing values can cause errors during training, so we need to know where they are.
print("Requirement 1c: Missing Data Evaluation")
print("----------------------------------------")

missing_data = data.isnull().sum()
missing_percentage = (missing_data / len(data)) * 100

missing_summary = pd.DataFrame({
    'Missing Values': missing_data,
    'Percentage': missing_percentage.round(2)
})

# Display only columns with missing values
print(missing_summary[missing_summary['Missing Values'] > 0])
print("----------------------------------------\n")


# -----------------------------------------------------------
# Requirement 1d: Visualizations
# -----------------------------------------------------------
print("Requirement 1d: Graphs and Visualizations")
# --- Visualizations: Understanding Theft Behavior Over Time ---

# 1. Theft frequency per year
# This bar chart shows how many bike thefts occurred each year.
# Helps identify trends such as increases or decreases over time.
plt.figure(figsize=(10, 6))
sns.countplot(x='OCC_YEAR', data=data, palette='viridis')
plt.title('Theft Frequency by Year')
plt.xlabel('Year')
plt.ylabel('Number of Thefts')
plt.show()

# 2. Theft distribution by hour of day
# This histogram shows what time of day bikes are stolen most.
# There are 24 bins (0–23), one for each hour.
plt.figure(figsize=(10, 6))
sns.histplot(data['OCC_HOUR'], bins=24, kde=True, color='blue')
plt.title('Theft Timing by Hour of Day')
plt.xlabel('Hour of Day (0-23)')
plt.ylabel('Frequency')
plt.show()

print("Key visualizations have been generated.")
print("----------------------------------------\n")


# ==============================================================================
# PART 2: DATA MODELLING (PREPARATION)
# ==============================================================================
print("===== PART 2: DATA MODELLING (PREPARATION) =====\n")

# Make a copy of the data so the original stays untouched
prep_data = data.copy()  

# -----------------------------------------------------------
# Requirement 2a: Handle Missing Data
# -----------------------------------------------------------

# For numeric columns: replace missing values with the median
# Median is used because it is stable and not affected by outliers
for col in ['BIKE_COST', 'BIKE_SPEED']:
    prep_data[col].fillna(prep_data[col].median(), inplace=True)

# For categorical (text) columns: replace missing with 'Unknown'
# This avoids guessing and keeps the data consistent
for col in ['BIKE_COLOUR', 'BIKE_MAKE', 'BIKE_MODEL']:
    prep_data[col].fillna('Unknown', inplace=True)

# Convert STATUS (text) into numeric values for ML
# STOLEN becomes 1, RECOVERED becomes 0
prep_data['STATUS'] = prep_data['STATUS'].map({'STOLEN': 1, 'RECOVERED': 0})
prep_data.dropna(subset=['STATUS'], inplace=True)  # remove invalid rows
prep_data['STATUS'] = prep_data['STATUS'].astype(int)

print("Requirement 2a: Missing data handled.")


# -----------------------------------------------------------
# Requirement 2a (continued): Encode Categorical Features
# -----------------------------------------------------------
# Identify all columns that contain text (object type)
categorical_cols = prep_data.select_dtypes(include=['object']).columns.tolist()

# Only encode columns that have fewer than 50 unique values
# Avoid encoding ID columns like EVENT_UNIQUE_ID
categorical_cols_to_encode = [
    col for col in categorical_cols
    if prep_data[col].nunique() < 50 and col not in ['EVENT_UNIQUE_ID']
]

# Apply One-Hot Encoding (turn categories into 0/1 columns)
# drop_first=True prevents multicollinearity
prep_data = pd.get_dummies(prep_data, columns=categorical_cols_to_encode, drop_first=True)

print("Requirement 2a: Categorical data transformed using one-hot encoding.")
print("----------------------------------------\n")


# -----------------------------------------------------------
# Requirement 2b: Select stable input features for inference
# -----------------------------------------------------------
# Use only the fields that the frontend actually collects.
# We intentionally exclude OBJECTID/x/y because:
# - OBJECTID is an identifier, not a real predictor.
# - x/y can be mismatched by coordinate system at inference time.
selected_features = [
    'OCC_YEAR',
    'OCC_DAY',
    'OCC_DOY',
    'OCC_HOUR',
    'REPORT_YEAR',
    'REPORT_DAY',
    'REPORT_DOY',
    'REPORT_HOUR',
    'BIKE_SPEED',
    'BIKE_COST',
    'LONG_WGS84',
    'LAT_WGS84',
]

# X = features (input data), y = target label
X = prep_data[selected_features].copy()
y = prep_data['STATUS'].copy()

# Fill any residual numeric gaps.
X = X.fillna(X.median(numeric_only=True))

print("Requirement 2b: Stable feature set selected for training/inference:")
print(selected_features)
print("----------------------------------------\n")


# -----------------------------------------------------------
# Requirement 2c: Train/Test Split
# -----------------------------------------------------------
# We split the data into:
# - 70% training data (to teach the model)
# - 30% testing data (to evaluate the model on new/unseen cases)
# The 'stratify=y' ensures both sets keep the same stolen/recovered ratio.
# 'random_state=42' makes the split reproducible.

X_train, X_test, y_train, y_test = train_test_split(
    X, 
    y, 
    test_size=0.3,            # 30% of the data used for testing
    random_state=42,          # ensures we get the same split every time
    stratify=y                # keeps class distribution balanced
)

print("Requirement 2c: Training and testing sets created.")
print(f"Training set shape: {X_train.shape}")
print(f"Testing set shape: {X_test.shape}")
print("----------------------------------------\n")


# -----------------------------------------------------------
# Requirement 2a (continued): Normalize Features
# -----------------------------------------------------------
# StandardScaler standardizes the data by:
# - Subtracting the mean (centers the data at 0)
# - Dividing by the standard deviation (scales the spread to 1)
# This helps models like Logistic Regression learn more effectively.

scaler = StandardScaler()

# Fit the scaler ONLY on the training data (important to avoid data leakage)
X_train_scaled = scaler.fit_transform(X_train)

# Apply the same scaling to the test data using the learned parameters
X_test_scaled = scaler.transform(X_test)

print("Requirement 2a: Data normalization complete using StandardScaler.")
print("----------------------------------------\n")


# -----------------------------------------------------------
# Requirement 2d: Inspect class imbalance (no synthetic resampling)
# -----------------------------------------------------------
print("Requirement 2d: Class distribution")
print("----------------------------------------")
print(y_train.value_counts(normalize=True))
print("\nUsing class-weighted models instead of SMOTE to avoid distorted probabilities.")
print("----------------------------------------\n")


# ==============================================================================
# PART 3: PREDICTIVE MODEL BUILDING
# ==============================================================================
print("===== PART 3: PREDICTIVE MODEL BUILDING =====\n")

# -----------------------------------------------------------
# Requirement 3a: Train Logistic Regression
# -----------------------------------------------------------
print("Training Logistic Regression model...")
# Logistic Regression is a classification algorithm that predicts
# the probability of a bike being STOLEN (1) or RECOVERED (0).
# It learns patterns from the training data and is good for
# finding linear relationships between features and the target.
log_reg_model = LogisticRegression(
    random_state=42,
    max_iter=3000,
    class_weight='balanced'
)

# Train (fit) the model on scaled training data.
log_reg_model.fit(X_train_scaled, y_train)
print("Logistic Regression model trained successfully.")

# -----------------------------------------------------------
# Requirement 3a: Train Random Forest Classifier
# -----------------------------------------------------------
print("Training Random Forest model...")
# Random Forest averages many trees to reduce sharp one-feature flips
# and produces more stable risk scores than a single tree.
rf_model = RandomForestClassifier(
    n_estimators=400,
    min_samples_leaf=40,
    class_weight='balanced_subsample',
    n_jobs=-1,
    random_state=42,
)

# Train the Random Forest on scaled training data.
rf_model.fit(X_train_scaled, y_train)

print("Random Forest model trained successfully.")
print("----------------------------------------\n")

# ==============================================================================
# PART 4: MODEL SCORING AND EVALUATION
# ==============================================================================
print("===== PART 4: MODEL SCORING AND EVALUATION =====\n")

# -----------------------------------------------------------
# Requirement 4a: Evaluate Logistic Regression
# -----------------------------------------------------------
print("Evaluating Logistic Regression Model...")
# Use the trained Logistic Regression model to predict labels (0/1)
# on the scaled test dataset.
y_pred_log = log_reg_model.predict(X_test_scaled)

# Predict probability scores instead of hard labels.
# [:, 1] extracts the probability of the bike being STOLEN (class 1).
y_prob_log = log_reg_model.predict_proba(X_test_scaled)[:, 1]

# Compute accuracy: percentage of correct predictions.
log_accuracy = accuracy_score(y_test, y_pred_log)

print("\nLogistic Regression Accuracy:", round(log_accuracy, 4))

# Print classification metrics:
# precision, recall, F1-score for both classes.
print("\nLogistic Regression Classification Report:")
print(classification_report(y_test, y_pred_log))

# Confusion matrix: shows correct vs incorrect predictions.
# Format:
# [[True Negatives, False Positives],
#  [False Negatives, True Positives]]
log_cm = confusion_matrix(y_test, y_pred_log)
print("Logistic Regression Confusion Matrix:")
print(log_cm)

# ROC curve data:
# fpr_log = False Positive Rate
# tpr_log = True Positive Rate
# These are used to construct the ROC chart.
fpr_log, tpr_log, _ = roc_curve(y_test, y_prob_log)

# AUC score (0.0–1.0): measures model’s overall performance.
auc_log = roc_auc_score(y_test, y_prob_log)
print(f"Logistic Regression AUC: {auc_log:.4f}")
brier_log = brier_score_loss(y_test, y_prob_log)
print(f"Logistic Regression Brier Score: {brier_log:.4f}")
print("----------------------------------------\n")

# -----------------------------------------------------------
# Requirement 4a: Evaluate Random Forest
# -----------------------------------------------------------
# Predict labels (0/1) for the test set using Random Forest.
y_pred_rf = rf_model.predict(X_test_scaled)

# Predict probabilities (needed for ROC curve).
y_prob_rf = rf_model.predict_proba(X_test_scaled)[:, 1]

# Compute accuracy for Random Forest.
rf_accuracy = accuracy_score(y_test, y_pred_rf)
print("\nRandom Forest Accuracy:", round(rf_accuracy, 4))

# Print precision, recall, and F1-score.
print("\nRandom Forest Classification Report:")
print(classification_report(y_test, y_pred_rf))

# Confusion matrix for Random Forest predictions.
rf_cm = confusion_matrix(y_test, y_pred_rf)
print("Random Forest Confusion Matrix:")
print(rf_cm)

# ROC curve values for Random Forest.
fpr_rf, tpr_rf, _ = roc_curve(y_test, y_prob_rf)

# AUC score for Random Forest.
auc_rf = roc_auc_score(y_test, y_prob_rf)
print(f"Random Forest AUC: {auc_rf:.4f}")
brier_rf = brier_score_loss(y_test, y_prob_rf)
print(f"Random Forest Brier Score: {brier_rf:.4f}")
print("----------------------------------------\n")


# -----------------------------------------------------------
# Requirement 4a: Plot ROC Curves Together
# -----------------------------------------------------------
# Create a comparison plot of both ROC curves.
plt.figure(figsize=(10, 6))
# Plot ROC curve for Logistic Regression.
plt.plot(fpr_log, tpr_log, label=f"Logistic Regression (AUC = {auc_log:.3f})")
# Plot ROC curve for Random Forest.
plt.plot(fpr_rf, tpr_rf, label=f"Random Forest (AUC = {auc_rf:.3f})")
# Baseline (random guessing) diagonal line for reference.
plt.plot([0, 1], [0, 1], 'k--')  # diagonal baseline

plt.title("ROC Curve Comparison: Logistic Regression vs Random Forest")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.show()


# -----------------------------------------------------------
# Requirement 4b: Select Best Model
# -----------------------------------------------------------
print("\n===== MODEL COMPARISON =====")

# Display metrics so the user can compare performance.
print(
    f"Logistic Regression: Accuracy={log_accuracy:.4f}, "
    f"AUC={auc_log:.4f}, Brier={brier_log:.4f}"
)
print(
    f"Random Forest     : Accuracy={rf_accuracy:.4f}, "
    f"AUC={auc_rf:.4f}, Brier={brier_rf:.4f}"
)

# Select best model by AUC, and break close ties with lower Brier score.
if (auc_log > auc_rf) or (abs(auc_log - auc_rf) < 1e-4 and brier_log < brier_rf):
    print("\nBEST MODEL → Logistic Regression")
    best_model = log_reg_model
else:
    print("\nBEST MODEL → Random Forest")
    best_model = rf_model

print("Model scoring and evaluation complete.")
print("----------------------------------------\n")

# ------------------------ SERIALIZATION STEP ------------------------

ARTIFACTS_DIR = BASE_DIR / "artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = ARTIFACTS_DIR / "bike_theft_best_model.pkl"
SCALER_PATH = ARTIFACTS_DIR / "bike_theft_scaler.pkl"
FEATURES_PATH = ARTIFACTS_DIR / "bike_theft_selected_features.npy"
METADATA_PATH = ARTIFACTS_DIR / "bike_theft_model_metadata.json"

# Save model + scaler using pickle
with open(MODEL_PATH, "wb") as f:
    pickle.dump(best_model, f)

with open(SCALER_PATH, "wb") as f:
    pickle.dump(scaler, f)

# Save selected feature names (for correct ordering later)
np.save(FEATURES_PATH, np.array(selected_features, dtype=object))

metadata = {
    "target_positive_label": "STOLEN",
    "target_negative_label": "RECOVERED",
    "target_definition": (
        "Risk score for a reported theft case being labeled STOLEN "
        "(not recovered) vs RECOVERED."
    ),
    "training_rows_total": int(len(y)),
    "positive_rate_overall": float(y.mean()),
    "negative_rate_overall": float(1 - y.mean()),
    "positive_rate_train_split": float(y_train.mean()),
    "negative_rate_train_split": float(1 - y_train.mean()),
    "selected_features": selected_features,
    "best_model_type": type(best_model).__name__,
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
}

with open(METADATA_PATH, "w", encoding="utf-8") as f:
    json.dump(metadata, f, indent=2)

print(f"Best model saved to: {MODEL_PATH}")
print(f"Scaler saved to    : {SCALER_PATH}")
print(f"Feature list saved : {FEATURES_PATH}")
print(f"Metadata saved to  : {METADATA_PATH}")
