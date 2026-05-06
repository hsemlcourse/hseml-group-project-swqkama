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

# Column names
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

# Encode target
encoder = LabelEncoder()
df["class"] = encoder.fit_transform(df["class"])

X = df.drop("class", axis=1)
y = df["class"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=RANDOM_STATE,
    stratify=y,
)

# Models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(random_state=RANDOM_STATE),
    "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
    "Extra Trees": ExtraTreesClassifier(random_state=RANDOM_STATE),
    "SVM": SVC(probability=True),
    "XGBoost": XGBClassifier(eval_metric="logloss"),
}

results = []

best_model = None
best_score = 0

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

# Save results
results_df = pd.DataFrame(results)
print(results_df)

# Save best model
joblib.dump(best_model, "models/best_model.pkl")

print("\nBest model saved successfully.")
