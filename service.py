# import numpy as np
# import bentoml
# from bentoml.io import NumpyNdarray
# import bentoml
# import mlflow
# from dagshub import dagshub_logger


# import dagshub
# dagshub.init(repo_owner='prince19998', repo_name='BentoML_Project_1', mlflow=True)

# mlflow.set_track_uri("https://dagshub.com/prince19998/BentoML_Project_1.mlflow")

# with dagshub_logger() as logger:
#     logger.log_metrics({"accuracy": 0.95}) 


# iris_clf_runner = bentoml.sklearn.get("iris_clf:latest").to_runner()

# svc = bentoml.Service("iris_classifier", runners=[iris_clf_runner])

# @svc.api(input=NumpyNdarray(), output=NumpyNdarray())
# def classify(input_series: np.ndarray) -> np.ndarray:
#     result = iris_clf_runner.predict.run(input_series)
#     return result

import numpy as np
import bentoml
from bentoml.io import NumpyNdarray
import mlflow
from dagshub import dagshub_logger

# Set the MLflow tracking URI for DagsHub
mlflow.set_tracking_uri("https://dagshub.com/prince19998/BentoML_Project_1.mlflow")

# Initialize the DagsHub logger
with dagshub_logger() as logger:
    logger.log_metrics({"accuracy": 0.95})  # Example of logging a metric

# Load the BentoML runner for the Iris classifier
iris_clf_runner = bentoml.sklearn.get("iris_clf:latest").to_runner()

# Create a BentoML service for the classifier
svc = bentoml.Service("iris_classifier", runners=[iris_clf_runner])

# Define the API endpoint for classification
@svc.api(input=NumpyNdarray(), output=NumpyNdarray())
def classify(input_series: np.ndarray) -> np.ndarray:
    result = iris_clf_runner.predict.run(input_series)
    return result
