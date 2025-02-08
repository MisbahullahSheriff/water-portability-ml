import dagshub
import mlflow

def mlflow_dagshub_setup():
    mlflow.set_tracking_uri("https://dagshub.com/MisbahullahSheriff/water-portability-ml.mlflow")
    dagshub.init(repo_owner='MisbahullahSheriff', repo_name='water-portability-ml', mlflow=True)