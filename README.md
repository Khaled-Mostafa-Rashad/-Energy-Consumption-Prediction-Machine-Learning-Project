# ⚡ Energy Consumption Prediction — Machine Learning Project

A regression project that predicts building/grid energy consumption (kWh) by comparing **Linear Regression**, **KNN Regressor**, and **Random Forest Regressor** across three candidate datasets, then tunes, evaluates, and deploys the strongest model.

## 📋 Table of Contents

- [Overview](#-overview)
- [Datasets](#-datasets)
- [Pipeline](#-pipeline)
- [Results](#-results)
- [Key Findings](#-key-findings)
- [Installation](#-installation)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Tech Stack](#-tech-stack)

## 🔍 Overview

This project answers a series of guided research questions:

1. Which dataset is most learnable / predictive?
2. Which ML model best predicts energy consumption?
3. How does hyperparameter tuning improve performance?
4. Which features matter most?
5. Does feature scaling affect model performance?
6. Does the final model generalize well, or does it overfit?

The notebook also ends with a small **interactive CLI tool** that takes user input and returns a predicted kWh value from the trained model.

## 📊 Datasets

Three candidate datasets were evaluated before selecting the best one:

| Dataset | Description | Verdict |
|---|---|---|
| `electricity_consumption_optimization_dataset` | Building-level energy data | ❌ Not learnable (R² ≈ 0) |
| `delhi_power_load_dataset1` | City-level power load data | ✅ Usable (R² ≈ 0.95) |
| `5.energy_consumption_grid` | Grid-level consumption data | ✅ **Best** (R² ≈ 0.96+) |

The grid-level dataset (`energy_consumption_grid.csv`) was selected as the final dataset based on consistently strong performance across all three model types.

## 🔄 Pipeline

```
Load & Clean Data
       │
       ▼
Domain-Knowledge Filtering (remove physically implausible values)
       │
       ▼
Feature Engineering (datetime → hour/day/month/year, building_age, etc.)
       │
       ▼
Encoding (Label Encoding for binary categories, One-Hot for the rest)
       │
       ▼
Correlation-Based Feature Selection
       │
       ▼
Train/Test Split + StandardScaler
       │
       ▼
Train Linear Regression, KNN, Random Forest
       │
       ▼
Hyperparameter Tuning (GridSearchCV / RandomizedSearchCV)
       │
       ▼
Model Comparison → Select Best Model
       │
       ▼
Feature Importance (ablation testing)
       │
       ▼
Overfitting Check (train R² vs test R²)
       │
       ▼
Interactive Prediction Tool
```

## 📈 Results

### Model comparison (final, tuned models)

| Model | MAE | MSE | R² |
|---|---|---|---|
| Linear Regression | 25.20 | 2733.38 | 0.9173 |
| KNN Regressor (k=35, manhattan) | 13.17 | 1182.02 | 0.9642 |
| **Random Forest** (tuned) | 13.81 | 1156.93 | **0.9650** 🏆 |

### Best hyperparameters found

**KNN Regressor**
```python
{'metric': 'manhattan', 'n_neighbors': 35}
```

**Random Forest Regressor**
```python
{'n_estimators': 200, 'max_depth': 10, 'min_samples_split': 5, 'min_samples_leaf': 4}
```

### Generalization check (train R² vs test R²)

| Model | Train R² | Test R² | Gap | Verdict |
|---|---|---|---|---|
| KNN Regressor | 0.9657 | 0.9642 | 0.0015 | ✅ Excellent — generalizes well |
| Random Forest | 0.9680 | 0.9650 | 0.0030 | ✅ Excellent — generalizes well |

## 💡 Key Findings

- **Scaling matters a lot for KNN, not at all for tree-based models.** Removing `StandardScaler` dropped KNN's R² from 0.96 to **0.49**, while Random Forest and Linear Regression were unaffected. This is because KNN relies on raw distance calculations (dominated by whichever feature has the largest numeric range), while Random Forest splits on thresholds that are scale-invariant.
- **`co2_emission_g` is the most important feature** — removing it drops R² from 0.9827 to 0.9607. `solar_generation_kwh`, by contrast, barely affects the result when removed.
- **Hyperparameter tuning gave modest but real gains**: Random Forest improved from R²=0.9576 → 0.9650, KNN from R²=0.9608 → 0.9634.
- **No overfitting detected** — train/test R² gaps under 0.003 for both top models.

## 🛠 Installation

```bash
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
pip install -r requirements.txt
```

### requirements.txt

```
pandas
numpy
seaborn
matplotlib
scikit-learn
scipy
```

## 🚀 Usage

1. Place the dataset CSV files inside a `Choosed/` folder in the project root.
2. Open the notebook:

```bash
jupyter notebook Machine_Learning_Project.ipynb
```

3. Run all cells top to bottom (**Kernel → Restart & Run All**).
4. When prompted `Do you want the visuals on? y/n`, type `n` for a faster run without plots, or `y` to see correlation heatmaps and distribution plots.
5. At the end of the notebook, the interactive prediction tool will ask for feature values and return a predicted energy consumption in kWh.

## 📁 Project Structure

```
.
├── Machine_Learning_Project.ipynb   # Main notebook
├── Choosed/
│   ├── electricity_consumption_optimization_dataset BULDING.csv
│   ├── delhi_power_load_dataset1 CITY.csv
│   └── 5.energy_consumption_grid.csv
├── requirements.txt
└── README.md
```

## 🧰 Tech Stack

- **Python 3**
- **pandas / numpy** — data manipulation
- **seaborn / matplotlib** — visualization
- **scikit-learn** — modeling, preprocessing, and hyperparameter tuning
  - `LinearRegression`, `KNeighborsRegressor`, `RandomForestRegressor`
  - `GridSearchCV`, `RandomizedSearchCV`
  - `StandardScaler`, `LabelEncoder`
- **scipy** — randomized search distributions

---

*This project was built as part of a Machine Learning coursework assignment.*
