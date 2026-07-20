# Feature Store and Data Management (Feast)

**Tags:** #mlops #feast #feature-store #data-engineering

## 🎯 Goal
Centralize and standardize machine learning feature definitions to ensure consistency between offline model training and online real-time inference, strictly preventing data leakage.

## 🛠 Technologies 
- **Feast (Feature Store)**: An open-source operational data system that manages and serves machine learning features to models in production.
- **Parquet / SQLite**: Used as the default offline storage layer and local registry during scaffolding and testing.

## 💻 Key commands 

### 1. Scaffold a Feature Repository
Initialize a new Feast project. This creates a standard directory structure including a feature_store.yaml configuration file.
```bash 
feast init feature_repo 
cd feature_repo/
```

### 2. Define Entities and Feature Views (Schema)
Features are defined declaratively in Python. An 'Entity' represents the primary key (e.g., a driver), and a 'FeatureView' groups related features mapped to that entity from a specific data source.
```python
# features.py
from datetime import timedelta
from feast import Entity, FeatureView, Field, FileSource
from feast.types import Float32, Int64

# Define the primary key used to join features with your dataset
driver = Entity(name="driver", join_keys=["driver_id"])

# Define the data source (Offline storage, e.g., a Parquet file, S3, or BigQuery)
driver_stats_source = FileSource(
    path="data/driver_stats.parquet",
    timestamp_field="event_timestamp",
    created_timestamp_column="created"
)

# Define the Feature View linking the entity, schema, and source
driver_stats_fv = FeatureView(
    name="driver_hourly_stats",
    entities=[driver],
    ttl=timedelta(days=1), # The time window to look back for a valid feature value
    schema=[
        Field(name="conv_rate", dtype=Float32),
        Field(name="acc_rate", dtype=Float32),
    ],
    source=driver_stats_source,
)
```

### 3. Apply the Infrastructure State
Parse the Python definitions and register them in the Feast registry (updates the state, usually tracked in a SQLite db or cloud bucket).
```bash
feast apply
```

### 4. Build a Training Set (Historical Features)
Retrieve point-in-time correct features for model training, appending them to a dataframe of labels and timestamps.
```python
import pandas as pd
from feast import FeatureStore

# Connect to the local feature store
fs = FeatureStore(repo_path=".")

# Target dataframe containing entities, timestamps, and target labels
entity_df = pd.DataFrame.from_dict({
    "driver_id": [1001, 1002],
    "event_timestamp": [
        pd.Timestamp("2026-07-10 10:00:00"),
        pd.Timestamp("2026-07-11 11:00:00")
    ],
    "target_label": [1, 0]
})

# Fetch point-in-time historical features to merge with the entity dataframe
training_data = fs.get_historical_features(
    entity_df=entity_df,
    features=[
        "driver_hourly_stats:conv_rate",
        "driver_hourly_stats:acc_rate"
    ]
).to_df()
```
### 5. Materialize Features to the Online Store
Offline storage (like Parquet or S3) is too slow for real-time model inference. Materialization scans the offline data and pushes only the latest feature values for each entity into a low-latency online store (like Redis, DynamoDB, or local SQLite).

```bash
# Sync the latest feature values from the offline store to the online store
# Using an ISO-8601 timestamp to define the end of the materialization window
feast materialize-incremental 2026-07-20T00:00:00
```
### 6. Read Features from the Online Store (Real-Time Inference)

Fetch the pre-computed latest feature values at low latency to feed into the model during a live prediction request. Unlike historical features, this does not require a timestamp—it just grabs the newest available data.
```python
from feast import FeatureStore

fs = FeatureStore(repo_path=".")

# Request the latest features for specific entities (e.g., driver_id = 1001)
online_features = fs.get_online_features(
    features=[
        "driver_hourly_stats:conv_rate",
        "driver_hourly_stats:acc_rate"
    ],
    entity_rows=[{"driver_id": 1001}]
).to_dict()

print(online_features)
# Output: {'driver_id': [1001], 'conv_rate': [0.55], 'acc_rate': [0.98]}
```
## 🧠 Conclusions and Problem Solutions

- **`FeastObjectNotFoundException` when fetching features**: You attempted to fetch features before the registry was aware of them. **Solution**: Always run `feast apply` in the terminal after creating or modifying your Python feature definitions (`features.py`) to sync the registry state before calling `get_historical_features()`.
- **Data Leakage during standard SQL joins**: When building training datasets using standard pandas or SQL joins, you might accidentally pull in feature values that were generated _after_ the target event occurred. **Solution**: Use Feast's `get_historical_features()`. It automatically performs a strict "point-in-time" join, looking only backward within the `ttl` (Time-To-Live) window relative to the `event_timestamp`, physically preventing future data from bleeding into the training set.
- **`get_online_features` returns `None` or missing values**: You successfully ran `feast apply` and offline data exists, but the online store returns empty arrays during inference. **Root cause**: The online store is physically empty because the data hasn't been synced, or the offline data's timestamps are older than the feature view's `ttl` (Time-To-Live). **Solution**: Run `feast materialize-incremental <current_date>` to populate the online store. If it still returns `None`, check if your offline data is stale and increase the `ttl` in the `FeatureView` definition.