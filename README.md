# human-personality-predictor

![Build](https://img.shields.io/badge/build-passing-brightgreen)
![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Issues](https://img.shields.io/github/issues/karoldziubak/human-personality-predictor)
![Forks](https://img.shields.io/github/forks/karoldziubak/human-personality-predictor)
![Stars](https://img.shields.io/github/stars/karoldziubak/human-personality-predictor)
![Last Commit](https://img.shields.io/github/last-commit/karoldziubak/human-personality-predictor)
![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)


🤖 **Human Personality Predictor** is an end-to-end machine learning project for classifying individuals as Introverts or Extroverts based on their social behavior, using structured tabular data.

---

## 📚 Table of Contents

- [human-personality-predictor](#human-personality-predictor)
  - [📚 Table of Contents](#-table-of-contents)
    - [🧭 Workflow Overview](#-workflow-overview)
    - [🚀 Main Features](#-main-features)
    - [📁 Project Structure](#-project-structure)
    - [📊 Data Description](#-data-description)
    - [🧠 Model Overview](#-model-overview)
    - [🎯 Project Goals](#-project-goals)
    - [🛠️ Installation](#️-installation)
    - [💡 Usage Examples](#-usage-examples)
    - [📄 Data Source](#-data-source)

---

### 🧭 Workflow Overview

| 🗂️ Raw Data      | 🧹 Preprocessing         | 🧠 Modeling         | 📊 Evaluation         | 📈 Visualization      |
|:----------------:|:-----------------------:|:------------------:|:---------------------:|:--------------------:|
| CSV, features, labels | Cleaning, encoding, splitting | ML, ensembles, DL | Metrics, ROC, PR      | Plots, reports       |

---

### 🚀 Main Features

- 📊 **Data Analysis & Visualization**: EDA, feature engineering, and clear visualizations
- 🧠 **Modeling**: Baselines, classical ML, ensembles, deep learning, and AutoML
- 🏆 **Evaluation**: Advanced metrics, confusion matrix, ROC/PR curves
- 🧩 **Modular Code**: Clean, reusable, and well-documented codebase
- 📓 **Jupyter Notebooks**: Ready-to-use for experiments and reproducibility
- 🛠️ **Production Workflow**: Scripts and structure for real-world ML projects

---

### 📁 Project Structure

```
human-personality-predictor/
│
├── data/                # Raw and processed datasets
├── notebooks/           # Jupyter notebooks for EDA, modeling, comparison
├── src/
│   ├── data_loader.py   # Data loading and preprocessing
│   └── models/          # All model classes and utilities
│       ├── base_model.py
│       ├── baseline_logistic_regression.py
│       ├── baseline_decision_tree.py
│       ├── ensemble_random_forest.py
│       ├── ensemble_xgboost.py
│       ├── ensemble_lightgbm.py
│       ├── ensemble_catboost.py
│       └── paramaters.py
├── requirements.txt     # Python dependencies
├── README.md            # Project documentation
└── ...                  # Other files
```

---

### 📊 Data Description

- **Source:** [Kaggle: Extrovert vs Introvert Behavior Data](https://www.kaggle.com/datasets/rakeshkapilavai/extrovert-vs-introvert-behavior-data/data)
- **Features:** Social behavior, communication patterns, preferences, etc.
- **Target:** Personality type (Introvert/Extrovert)
- **Preprocessing:** Handling missing values, encoding categorical variables, feature scaling, train-test split.

---

### 🧠 Model Overview

- **Baselines:** Logistic Regression, Decision Tree
- **Classical ML:** SVM, KNN, Perceptron (planned)
- **Ensembles:** Random Forest, XGBoost, LightGBM, CatBoost
- **Deep Learning:** Planned (PyTorch)
- **AutoML:** Planned
- All models support hyperparameter tuning and unified evaluation.

---

### 🎯 Project Goals

- Benchmark a wide range of ML techniques on a real-world personality dataset
- Provide clear, well-documented code for educational and practical use
- Serve as a template for similar classification problems in tabular data

---

### 🛠️ Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/karoldziubak/human-personality-predictor.git
   cd human-personality-predictor
   ```
2. (Recommended) Create and activate a virtual environment:
   ```bash
   python -m venv venv
   # Windows:
   venv\Scripts\activate
   # Linux/macOS:
   source venv/bin/activate
   ```
3. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

---

### 💡 Usage Examples

**Run a notebook:**

Open Jupyter and start experimenting:
```bash
jupyter notebook notebooks/model_comparison.ipynb
```

**Use DataLoader in Python:**

```python
from src.data_loader import DataLoader
loader = DataLoader()
X_train, X_test, y_train, y_test = loader.get_data_train_test()
```

**Train and evaluate a model:**

```python
from src.models.ensemble_random_forest import RandomForestModel
model = RandomForestModel()
model.train(X_train, y_train)
results = model.evaluate(X_test, y_test, verbose=True)
print(results)
```

---

### 📄 Data Source

**Kaggle:** [Extrovert vs Introvert Behavior Data](https://www.kaggle.com/datasets/rakeshkapilavai/extrovert-vs-introvert-behavior-data/data)