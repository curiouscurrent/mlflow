1) Install mlflow
```
pip install mlflow
```
2) To check the User Interface
```
mlflow ui
```
### First run (Logged metrics and params using MLflow)
![alt text](images/image.png)
4) Find the mlflow artifact scheme with 
```
mlflow.get_tracking_uri()
```
5) Set tracking URI for artifact scheme, server running on this. This is initially in sqlite format, convert it to http/https format. 
```
mlflow.set_tracking_uri("http://127.0.0.1:5000/")
```

6) Explicitly set the experiment name to ensure that the run is logged under the correct experiment
```
mlflow.set_experiment("<exp-name>")
```
or 

set the exp id inside the mlrun code
```
with mlflow.start_run(experiment_id=):
```

7) Set remote tracking server via Dagshub
```
pip install dagshub
```

8) Autolog logs everything like params(all possible params of the model), metrics, artifacts, models except the script
```
mlflow.autolog()
```


# OUTPUTS

### Logged Artifacts using MLflow
![alt text](images/image-1.png)

### Experiment vs Run (Visual understanding)
![alt text](images/image-2.png)

### Logged the model along with its dependencies
![alt text](images/image-3.png)

### Added tags 
![alt text](images/image-4.png)

### MLflow Server Architecture
![alt text](images/image-5.png)

### Dagshub connected
![alt text](images/image-7.png)

### Remote Mlflow server setup using Dagshub
![alt text](images/image-6.png)

### Logged artifacts, params, datasets, models using Autologging
![alt text](images/image-8.png)
![alt text](images/image-9.png)



### Hyperparameter tuning using Mlflow
![alt text](images/image-10.png)
![alt text](images/image-11.png)

### Stages of Model Registry
![alt text](images/image-12.png)

### Model registry performed.
![alt text](images/image-13.png)