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
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

# initialize mlflow and dagshub
mlflow.set_tracking_uri(
    "https://dagshub.com/MisbahullahSheriff/water-portability-ml.mlflow"
)
dagshub.init(
    repo_owner='MisbahullahSheriff', repo_name='water-portability-ml', mlflow=True
)

# data
df = pd.read_excel(
    "C:/python-programs/mlops-tutorials/water-portability-ml/data/raw/water_potability.xlsx"
)
X = df.iloc[:, :-1]
y = df.iloc[:, -1].copy()
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, shuffle=True, random_state=42
)

# preprocessing
preprocessor = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])
X_train_pre = preprocessor.fit_transform(X_train)
X_test_pre = preprocessor.transform(X_test)

models = {
    "knn": KNeighborsClassifier(),
    "naive_bayes": GaussianNB(),
    "logistic_regression": LogisticRegression(),
    "svm": SVC(),
    "decision_tree": DecisionTreeClassifier(),
    "random_forest": RandomForestClassifier()
}

mlflow.set_experiment("Impute Mean and Standardization")
for model_name, model in models.items():
    save_model_name = f"{model_name}.pkl"
    with mlflow.start_run(run_name=model_name):
        # train model
        model.fit(X_train_pre, y_train)
        pickle.dump(model, open(save_model_name, "wb"))

        # load model and predict
        saved_model = pickle.load(open(save_model_name, "rb"))
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
        plt.xlabel("Predicted Labels")
        plt.ylabel("Actual Labels")
        plt.title(f"Confusion Matrix - {model_name.replace('_', ' ').title()}")
        plt.savefig(f"confusion_matrix_{model_name}.png")

        mlflow.log_metrics(metrics={
            "accuracy": acc,
            "f1-score": f1_val,
            "precision": precision,
            "recall": recall
        })
        mlflow.log_artifact(__file__)
        mlflow.log_artifact(f"confusion_matrix_{model_name}.png")
        mlflow.sklearn.log_model(model, model_name.replace("_", " ").title())