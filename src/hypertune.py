from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
import pandas as pd
import mlflow
import dagshub

## Remote server setup below using Dagshub
# -------------------------------------------------------------------------------
# Set MLflow tracking URI
mlflow.set_tracking_uri("https://dagshub.com/curiouscurrent/mlflow.mlflow")

# Mlflow tracking setup (dagshub mlflow integration)
dagshub.init(repo_owner='curiouscurrent', repo_name='mlflow', mlflow=True)
# ------------------------------------------------------------------------------

# Load the Breast Cancer dataset
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name='target')

# Splitting into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Creating the RandomForestClassifier model
rf = RandomForestClassifier(random_state=42)

# Defining the parameter grid for GridSearchCV
param_grid = {
    'n_estimators': [10, 50, 100],
    'max_depth': [None, 10, 20, 30]
}

# Applying GridSearchCV
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)

# # Run without MLflow from here
# grid_search.fit(X_train, y_train)

# Displaying the best params and best score
# best_params = grid_search.best_params_
# best_score = grid_search.best_score_

# print(best_params)
# print(best_score)
# cv_df = pd.DataFrame(grid_search.cv_results_)
# print(cv_df.columns)
'''
Inside grid_search.cv_results_, we have the following keys:

Index(['mean_fit_time', 'std_fit_time', 'mean_score_time', 'std_score_time',
       'param_max_depth', 'param_n_estimators', 'params', 'split0_test_score',
       'param_max_depth', 'param_n_estimators', 'params', 'split0_test_score',
       'split1_test_score', 'split2_test_score', 'split3_test_score',
       'split4_test_score', 'mean_test_score', 'std_test_score',
       'rank_test_score'],
      dtype='object')
'''

mlflow.set_experiment('breast-cancer-hyperparameter-tuning')

with mlflow.start_run() as parent:
    grid_search.fit(X_train, y_train)

    # log all the child runs
    for i in range(len(grid_search.cv_results_['params'])):
        # In params, we have all the combinations of max_depth and n_estimators that we tried in GridSearchCV.
        # We will log each of these combinations as a child run under the parent run, along with the corresponding mean_test_score for that combination.

        with mlflow.start_run(nested=True) as child:
            mlflow.log_params(grid_search.cv_results_["params"][i])
            mlflow.log_metric("accuracy", grid_search.cv_results_["mean_test_score"][i])

    # Displaying the best parameters and the best score
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    # Log params
    mlflow.log_params(best_params)

    # Log metrics
    mlflow.log_metric("accuracy", best_score)

    # Log training data
    train_df = X_train.copy()
    train_df['target'] = y_train

    train_df = mlflow.data.from_pandas(train_df)
    mlflow.log_input(train_df, "training")

    # Log test data
      #Input test data being logged to MLflow using mlflow.data.from_pandas to convert the pandas 
      # DataFrame into a format that can be logged as an input artifact in MLflow.

    test_df = X_test.copy()
    test_df['target'] = y_test

    test_df = mlflow.data.from_pandas(test_df)
    mlflow.log_input(test_df, "testing")

    # Log source code
    mlflow.log_artifact(__file__)

    # Log the best model
    mlflow.sklearn.log_model(grid_search.best_estimator_, "random_forest")

    # Set tags
    mlflow.set_tag("author", "Anusha K")

    print(best_params)
    print(best_score)
