import mlflow

print(mlflow.get_tracking_uri())

# after setting the tracking URI, we can check it again to confirm that it has been set correctly
mlflow.set_tracking_uri("http://127.0.0.1:5000/")
print(mlflow.get_tracking_uri())