# import bentoml
# import mlflow
# from dagshub import dagshub_logger

# import logging
# import sys
# import warnings
# from urllib.parse import urlparse

# import mlflow
# import mlflow.sklearn
# from mlflow.models import infer_signature

# logging.basicConfig(level=logging.WARN)
# logger = logging.getLogger(__name__)

# import dagshub
# dagshub.init(repo_owner='prince19998', repo_name='BentoML_Project_1', mlflow=True)

# mlflow.set_track_uri("https://dagshub.com/prince19998/BentoML_Project_1.mlflow")

# from dagshub import dagshub_logger
# logger = dagshub_logger(repo_owner="prince19998", repo_name="BentoML_Project_1")
# logger.init(mlflow=True)


# iris_clf_runner = bentoml.sklearn.get("iris_clf:latest").to_runner()
# iris_clf_runner.init_local()
# print(iris_clf_runner.predict.run([[5.9, 3., 5.1, 1.8]]))

import bentoml
import mlflow
from dagshub import dagshub_logger

# Set the MLflow tracking URI
mlflow.set_tracking_uri("https://dagshub.com/prince19998/BentoML_Project_1.mlflow")

# Initialize the DagsHub logger
logger = dagshub_logger(repo_owner="prince19998", repo_name="BentoML_Project_1")
logger.init(mlflow=True)  # This will enable DagsHub's MLflow tracking

# Load the BentoML runner for the model
iris_clf_runner = bentoml.sklearn.get("iris_clf:latest").to_runner()
iris_clf_runner.init_local()

# Perform a prediction
print(iris_clf_runner.predict.run([[5.9, 3., 5.1, 1.8]]))
