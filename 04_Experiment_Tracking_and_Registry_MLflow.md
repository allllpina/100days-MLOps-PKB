# Experiment Tracking and Model Registry (MLflow)

**Tags:** #mlops #mlflow #experiment-tracking #model-registry

## 🎯 Goal
Centralize machine learning experiment tracking, manage model lifecycles, and serve models systematically using a unified MLflow server.

## 🛠 Technologies 
- **MLflow**: A platform to streamline machine learning development, including tracking experiments, packaging code into reproducible runs, and managing model deployment.
- **SQLite**: A lightweight relational database used as the backend store for logging MLflow experiment parameters, metrics, and metadata.

## 💻 Key commands 

### 1. Create necessary parent directories for the backend store and artifacts to prevent startup crashes
```bash 
# Directory Initialization
mkdir -p /root/code/mlflow-backend /root/code/mlflow-artifacts/
```

### 2. Launch the server on all interfaces (0.0.0.0), specifying SQLite backend, artifact root, and allowing all CORS/hosts for proxy compatibility. Run with nohup to survive terminal closure.
```bash
# Start MLflow Tracking Server (Background)
nohup mlflow server

--host 0.0.0.0

--port 5000

--backend-store-uri sqlite:////root/code/mlflow-backend/mlflow.db

--default-artifact-root /root/code/mlflow-artifacts/

--cors-allowed-origins ''

--allowed-hosts '' > mlflow_server.log 2>&1 &
```

### 3. Connect to the tracking server and log hyperparameters, metrics, and the model artifact inside a run context.
```python
import mlflow
import mlflow.sklearn

# Connect to the active tracking server
mlflow.set_tracking_uri("http://localhost:5000")

params = {"n_estimators": 100, "max_depth": 5, "random_state": 42}
metrics = {"accuracy": 0.92, "f1_score": 0.89}
# model = ... (your trained sklearn model)

# Start an MLflow run using a context manager
with mlflow.start_run():
    # Log dictionaries of hyperparameters and metrics
    mlflow.log_params(params)
    mlflow.log_metrics(metrics)
    
    # Log the trained model object as an artifact in the "model" directory
    mlflow.sklearn.log_model(model, "model")
```

### 4. Automatic Logging (Autologging) 
Instead of manually writing `log_param` and `log_metric` for every value, MLflow can automatically hook into supported machine learning libraries (like Scikit-learn, XGBoost, TensorFlow, PyTorch, LightGBM). It works by patching the library's training functions (e.g., `.fit()`) behind the scenes to capture parameters, metrics, tags, and even the model artifact itself automatically. 
```python 
import mlflow 
import mlflow.sklearn 

mlflow.set_tracking_uri("http://localhost:5000")

# 1. Universal autologging (tries to detect and log for all supported libraries)
mlflow.autolog()

# 2. Alternatively, use flavor-specific autologging for precise control 
mlflow.sklearn.autolog(log_models=True, log_input_examples=True, log_datasets=False)

with mlflow.start_run(): 
	# When .fit() is called, MLflow intercepts it and automatically logs 
	# the model parameters, default metrics, and the serialized model artifact. 
	model.fit(X_train, y_train)
	
	# You can still manually log custom business metrics alongside autologging
	mlflow.log_metric("custom_business_score", 0.95)
```
### 5. Model Lifecycle: Logged vs. Registered Models
Understanding the strict architectural boundary between Data Science experimentation and DevOps production deployment.

#### 🧠 Concept Breakdown: The 3 Stages of Model Handoff

**1. Logged Model (The "Laboratory")**
* **What it is:** When you run `mlflow.sklearn.log_model()` or use autologging, MLflow creates an artifact folder inside a specific Run.
* **Contents:** It contains the serialized model (e.g., `model.pkl`), environment requirements (`requirements.txt`, `conda.yaml`), and the `MLmodel` manifest.
* **Purpose:** It is strictly an archive of an experiment. It sits alongside hundreds of other failed or successful iterations in the "Experiments" tab.

**2. Registered Model (The "Showcase" & Contract)**
* **What it is:** You cannot deploy a model using a raw path like `/runs/a1b2c3/artifacts/model`—it is unreliable. Instead, you "Register" the model.
* **How it works:** Registering creates a top-level logical entity (e.g., `fraud-detector`). It does *not* copy the model files. It simply creates a **Version** (e.g., `v1`) that acts as a pointer back to the Logged Model's artifacts. 
* **Purpose:** It acts as a clean, organized catalog of only the best, production-ready models.

**3. Separation of Concerns (DS vs. DevOps)**
The Model Registry serves as the official API/Contract between teams:
* **Data Scientist (DS):** Works entirely in the *Runs/Experiments* view. Analyzes metrics. When they find the best model, they register it and assign a semantic tag/alias (e.g., `@champion`). They guarantee the model's predictive quality.
* **DevOps / Platform Engineer:** *Never* opens the Experiments tab. They write CI/CD pipelines that blindly request the model using its name and alias (`models:/fraud-detector@champion`). They don't care about hyperparameters; they only care about wrapping the downloaded artifacts into a Docker container and serving it.

#### 💻 Key commands
### 1. Registering a model from an existing run
Using the Python API to register a model, creating a new version in the Model Registry.
```python
import mlflow

# The URI of the logged model from a specific run
logged_model_uri = "runs:/<run_id>/model"

# Register the model (creates the entity if it doesn't exist, or adds a new version)
mlflow.register_model(model_uri=logged_model_uri, name="fraud-detector")
```
### 2. Assigning an Alias (The "Champion" tag)
The Data Scientist assigns an alias to indicate the model's readiness state.

```Python
from mlflow import MlflowClient
client = MlflowClient()

# Point the "champion" alias to version 1 of the fraud-detector
client.set_registered_model_alias("fraud-detector", "champion", "1")
```
### 3. DevOps / CI/CD Retrieval
The Platform Engineer loads the model in the production environment using the stable alias, completely ignoring Run IDs.
```python
import mlflow.pyfunc

# Load the model directly from the registry using its semantic alias
prod_model = mlflow.pyfunc.load_model("models:/fraud-detector@champion")
```
## 🧠 Conclusions and Problem Solutions

- **MLflow server aborts on startup**: The specified backend directory does not exist. **Solution**: Always explicitly create the required parent directories before launching the server process.
- **UI dashboard inaccessible or proxy rejects requests**: The server restricts origins or host headers by default. **Solution**: Pass `--cors-allowed-origins '*'` and `--allowed-hosts '*'` flags to accept requests routed through a proxy environment.
- **Server process dies when the terminal closes**: The command was run in the foreground. **Solution**: Use `nohup ... &` to detach the process from the terminal session and keep it running in the background.
- **Runs are left hanging in "RUNNING" state**: The script crashed before `mlflow.end_run()` was called. **Solution**: Always use the context manager `with mlflow.start_run():`. It automatically handles closing the run (even if an exception occurs) and prevents orphaned runs in the UI.
- **Custom metrics are missing from the run**: `mlflow.autolog()` only captures standard metrics defined by the framework's default evaluation behavior (e.g., accuracy for sklearn classifiers). **Solution**: Compute your custom metrics manually and log them using `mlflow.log_metric()` inside the same `start_run()` context alongside the autologged `.fit()` call.
- **Autologging creates too much noise or uploads massive datasets**: Universal `mlflow.autolog()` logs everything by default, including training datasets if supported, which can bloat the artifact store. **Solution**: Use the flavor-specific function (e.g., `mlflow.sklearn.autolog(log_datasets=False)`) and pass specific boolean flags to disable logging for datasets or models.
- **Hardcoded Run IDs break CI/CD pipelines**: A deployment script fails because a Run ID was deleted or hardcoded as `/runs/abc/model`. **Solution**: Never use Run URIs in production code. Always register the model and use semantic aliases (`models:/model_name@champion`) so the data science team can update the underlying version without breaking the DevOps pipeline.