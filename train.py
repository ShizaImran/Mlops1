import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load the iris dataset
data = load_iris()
X = data.data
y = data.target

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define a function to train the model
def train_model(n_estimators, max_depth):
    with mlflow.start_run():
        # Initialize the model with the provided hyperparameters
        model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth)
        # Fit the model to the training data
        model.fit(X_train, y_train)
        # Log the model and parameters to MLflow
        mlflow.sklearn.log_model(model, "random_forest_model")
        mlflow.log_params({"n_estimators": n_estimators, "max_depth": max_depth})
        # Evaluate the model
        accuracy = model.score(X_test, y_test)
        # Log the accuracy metric
        mlflow.log_metric("accuracy", accuracy)

# Train the model with different hyperparameters
train_model(n_estimators=100, max_depth=5)
train_model(n_estimators=200, max_depth=10)
