import dagshub
import mlflow
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

mlflow.set_tracking_uri(
    "https://dagshub.com/MisbahullahSheriff/water-portability-ml.mlflow"
)
dagshub.init(
    repo_owner='MisbahullahSheriff', repo_name='water-portability-ml', mlflow=True
)

df = pd.read_excel(
    "C:/python-programs/mlops-tutorials/water-portability-ml/data/raw/water_potability.xlsx"
)
X = df.iloc[:, :-1]
y = df.iloc[:, -1].copy()

mlflow.set_experiment("Exp1")
with mlflow.start_run(run_name='baseline-rf'):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, shuffle=True, random_state=42
    )

    preprocessor = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    X_train_pre = preprocessor.fit_transform(X_train)
    X_test_pre = preprocessor.transform(X_test)

    clf = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=7)
    clf.fit(X_train_pre, y_train)
    pickle.dump(clf, open("baseline-model.pkl", "wb"))

    saved_model = pickle.load(open("baseline-model.pkl", "rb"))
    y_pred = saved_model.predict(X_test_pre)

    # metrics
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    f1_val = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    plt.figure(figsize=(8, 5))
    sns.heatmap(
        data=cm, cmap="Blues", annot=True, linewidths=1.5,
        linecolor="black", cbar=True, fmt=".0g",
        xticklabels=['predicted-false', 'predicted-true'],
        yticklabels=['actual-false', 'actual-true']
    )
    plt.savefig("confusion-matrix.png")

    mlflow.log_metrics(metrics={
        "accuracy": acc,
        "f1-score": f1_val,
        "precision": precision,
        "recall": recall
    })
    mlflow.log_params(params={
        "n_estimators": clf.n_estimators,
        "max_depth": clf.max_depth
    })
    mlflow.log_artifact(__file__)
    mlflow.log_artifact("confusion-matrix.png")
    mlflow.sklearn.log_model(clf, "Random Forest")

    print(f"Accuracy: {acc}")
    print(f"F1 Score: {f1_val}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")

    



