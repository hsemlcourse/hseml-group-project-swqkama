# Training and comparing ML models for MAGIC Gamma Telescope dataset

import joblib
import pandas as pd

from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

from xgboost import XGBClassifier


RANDOM_STATE = 42


# Load dataset
df = pd.read_csv("data/magic04.data")

# Column names from MAGIC Gamma Telescope dataset
df.columns = [
    "fLength",
    "fWidth",
    "fSize",
    "fConc",
    "fConc1",
    "fAsym",
    "fM3Long",
    "fM3Trans",
    "fAlpha",
    "fDist",
    "class",
]

# Encode target variable
encoder = LabelEncoder()
df["class"] = encoder.fit_transform(df["class"])

X = df.drop("class", axis=1)
y = df["class"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=RANDOM_STATE,
    stratify=y,
)

# Models for comparison
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(random_state=RANDOM_STATE),
    "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
    "Extra Trees": ExtraTreesClassifier(random_state=RANDOM_STATE),
    "SVM": SVC(probability=True, random_state=RANDOM_STATE),
    "XGBoost": XGBClassifier(eval_metric="logloss", random_state=RANDOM_STATE),
}

results = []
best_model = None
best_score = 0


# Train different models and compare metrics
for name, model in models.items():
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)

    results.append(
        {
            "Model": name,
            "Accuracy": accuracy,
            "F1-score": f1,
            "ROC-AUC": roc_auc,
        }
    )

    if roc_auc > best_score:
        best_score = roc_auc
        best_model = model


# Simple hyperparameter tuning for Random Forest
param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [5, 10, None],
}

grid_search = GridSearchCV(
    RandomForestClassifier(random_state=RANDOM_STATE),
    param_grid,
    cv=3,
    scoring="roc_auc",
)

grid_search.fit(X_train, y_train)

tuned_model = grid_search.best_estimator_

y_pred = tuned_model.predict(X_test)
y_proba = tuned_model.predict_proba(X_test)[:, 1]

tuned_accuracy = accuracy_score(y_test, y_pred)
tuned_f1 = f1_score(y_test, y_pred)
tuned_roc_auc = roc_auc_score(y_test, y_proba)

results.append(
    {
        "Model": "Random Forest tuned",
        "Accuracy": tuned_accuracy,
        "F1-score": tuned_f1,
        "ROC-AUC": tuned_roc_auc,
    }
)

if tuned_roc_auc > best_score:
    best_score = tuned_roc_auc
    best_model = tuned_model

print("Best RF parameters:", grid_search.best_params_)
print("Best RF CV ROC-AUC:", grid_search.best_score_)

# Save experiment results
results_df = pd.DataFrame(results)
print(results_df.sort_values(by="ROC-AUC", ascending=False))

# Save best model
joblib.dump(best_model, "models/best_model.pkl")

print("\nBest model saved successfully.")
