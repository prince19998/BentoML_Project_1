import numpy as np
import bentoml
from bentoml.io import NumpyNdarray

import dagshub
dagshub.init(repo_owner='prince19998', repo_name='BentoML_Project_1', mlflow=True)

mlflow.set_track_uri("https://dagshub.com/prince19998/BentoML_Project_1.mlflow")



iris_clf_runner = bentoml.sklearn.get("iris_clf:latest").to_runner()

svc = bentoml.Service("iris_classifier", runners=[iris_clf_runner])

@svc.api(input=NumpyNdarray(), output=NumpyNdarray())
def classify(input_series: np.ndarray) -> np.ndarray:
    result = iris_clf_runner.predict.run(input_series)
    return result