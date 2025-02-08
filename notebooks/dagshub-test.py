import dagshub
import mlflow

mlflow.set_tracking_uri("https://dagshub.com/MisbahullahSheriff/water-portability-ml.mlflow")

dagshub.init(repo_owner='MisbahullahSheriff', repo_name='water-portability-ml', mlflow=True)

with mlflow.start_run():
  mlflow.log_param('parameter name', 'value')
  mlflow.log_metric('metric name', 1)