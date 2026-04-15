1) Install mlflow
```
pip install mlflow
```
2) To check the User Interface
```
mlflow ui
```
### First run (Logged metrics and params using MLflow)
![alt text](image.png)
4) Find the mlflow artifact scheme with 
```
mlflow.get_tracking_uri()
```
5) Set tracking URI for artifact scheme, server running on this. This is initially in sqlite format, convert it to http/https format. 
```
mlflow.set_tracking_uri("http://127.0.0.1:5000/")
```
### Logged Artifacts using MLflow
![alt text](image-1.png)