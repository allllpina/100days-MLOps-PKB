# Python Environment & Dependency Management **Tags:** #mlops #uv #venv #jupyter #dependencies
## 🎯 Goal
To ensure complete isolation of an ML project's dependencies, fast package installation? and precise version control (reproducibility) to avoid conflicts across different environments.

## 🛠 Technologies 
- **`venv`**: Standard environment isolation. 
- **`uv`**: An extremely fast package manager (written in Rust) that replaces `pip` and `pip-tools`. 
- **`ipykernel`**: Integration of isolated environment with Jupyter.
- 
## 💻 Key commands 

### 1. Virtual environment management
```bash 
# Creation of environment
python3 -m venv .venv 
# Activation
source .venv/bin/activate
# Deactivation
deactivate
```

### 2. Working with Dependencies Using `uv`
Instead of a direct `pip install`, an approach that compiles dependencies is used to create strict lock files.
```bash
# 1. Create an exact lock file (`requirements.txt`) from high-level requirements (`requirements.in`)
uv pip compile requirements.in -o requirements.txt

# 2. Synchronizing the environment (installs exactly what’s in the lock file and removes anything extra)
uv pip sync requirements.txt
```

### 3. Setting up the Jupyter server
To make Jupyter use the packages in `.venv` instead of the global ones:
```bash
uv pip install jupyter ipykernel
python -m ipykernel install --user --name=mlops-env --display-name="Python (MLOps)"
jupyter notebook --port 8888
```

## 🧠 Conclusions and Problem Solutions

- **Corrupted lock-file:** Caused by a conflict of transitive dependencies. Solution: Check the `uv` error log, explicitly specify compatible versions of the base packages in `requirements.in`, and run `uv pip compile` again.
- **Global Scope**: Never run Jupyter globally for a local project; always register the environment kernel.