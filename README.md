## BentoML Pratict Project 1


import dagshub
dagshub.init(repo_owner='prince19998', repo_name='BentoML_Project_1', mlflow=True)

import mlflow
with mlflow.start_run():
  mlflow.log_param('parameter name', 'value')
  mlflow.log_metric('metric name', 1)