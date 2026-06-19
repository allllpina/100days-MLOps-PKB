# Project Architecture and Automation

**Tags:** #mlops #cookiecutter #makefile #pre-commit #packaging

## 🎯 Goal
Standardize the directory structure for ML projects, automate routine tasks, and ensure automatic code quality checks before pushing to the version control system

## 🛠 Technologies 
- **Cookiecutter**: Generates the project directory structure from pre-defined templates to ensure reproducibility.
- **Makefile**: Automates ML workflows (like training, cleaning, or installing) by creating aliases for complex console commands.
- **Pre-commit**: Runs hooks (linters and formatters) automatically before a `git commit` to enforce code quality.
- **Build / Setuptools**: Packages local ML code into an installable Python distribution (`.whl` or `.tar.gz`).

## 💻 Key commands 

### 1. Project Templating (Cookiecutter)
```bash 
# Install cookiecutter and generate a project from a standard data science template
pip install cookiecutter
cookiecutter https://github.com/drivendata/cookiecutter-data-science
```

### 2. Workflow Automation (Makefile)
```Makefile
# Example of a Makefile structure (requires Tab indentation, not spaces)
.PHONY: install train clean

install:
	uv pip sync requirements.txt

train:
	python src/models/train_model.py

clean:
	find . -type f -name "*.py[co]" -delete
```
```bash
# Execute a specific target from the Makefile
make install
```

### 3. Code Quality Control (Pre-commit)
```bash
# Install the pre-commit hooks into your local .git/hooks directory
pre-commit install

# Manually run all hooks against all files (useful for initial setup)
pre-commit run --all-files
```

### 4. ML Project Packaging
```bash
# Build the project into an installable package (requires pyproject.toml or setup.py) 
python -m build
```

## 🧠 Conclusions and Problem Solutions

- **`Makefile: *** missing separator. Stop.` error**: Makefiles strictly require Tab characters for indentation under targets, but the text editor used spaces. **Solution**: Change your editor settings to use Tabs instead of spaces specifically for the `Makefile`, and replace the leading spaces before the commands with a single Tab.
- **`git commit` is blocked by pre-commit hooks**: Linters (like Ruff or Flake8) or formatters (like Black) found code quality issues or formatting inconsistencies. **Solution**: Check the pre-commit console output. Formatters usually auto-fix the files automatically, so you just need to `git add` the modified files and run `git commit` again. For structural linting errors, manually fix the code as suggested by the hook output, `git add` the changes, and commit.
